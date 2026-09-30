// C++17 port of markout.lob.fills.simulate_python. See queue_sim.hpp for the algorithm;
// comments here point at the matching step of the Python reference.
#include "queue_sim.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace markout {
namespace {

using Level = std::vector<std::size_t>;          // live orders at one price, activation order
using SideBook = std::map<std::int64_t, Level>;  // price -> level; ordered, so "every price
                                                 // beyond p" is a contiguous range

// A set of order ids kept as a sorted vector. Nasdaq assigns order reference numbers in
// increasing order, so inserts are almost always a push_back; it behaves exactly like the
// Python set it replaces (no duplicates, membership, removal) without a heap node per id.
class IdSet {
  public:
    void insert(std::int64_t id) {
        if (ids_.empty() || id > ids_.back()) {
            ids_.push_back(id);
            return;
        }
        auto it = std::lower_bound(ids_.begin(), ids_.end(), id);
        if (it == ids_.end() || *it != id) ids_.insert(it, id);
    }
    bool contains(std::int64_t id) const {
        return std::binary_search(ids_.begin(), ids_.end(), id);
    }
    void erase(std::int64_t id) {
        auto it = std::lower_bound(ids_.begin(), ids_.end(), id);
        if (it != ids_.end() && *it == id) ids_.erase(it);
    }
    void release() { std::vector<std::int64_t>().swap(ids_); }

  private:
    std::vector<std::int64_t> ids_;
};

std::int64_t depth_at(const std::int64_t* prices, const std::int64_t* sizes, std::size_t levels,
                      std::size_t n, std::int64_t price) {
    const std::size_t row = n * levels;
    for (std::size_t k = 0; k < levels; ++k) {
        if (prices[row + k] == price) return sizes[row + k];
    }
    return 0;
}

void validate(const Messages& msg, const Book& book, const Orders& ord) {
    if (book.n != msg.n) throw std::invalid_argument("book and messages must be row-aligned");
    if (book.levels == 0) throw std::invalid_argument("book needs at least one level");
    for (std::size_t j = 0; j < ord.k; ++j) {
        if (ord.place_idx[j] < 0 || static_cast<std::size_t>(ord.place_idx[j]) >= msg.n)
            throw std::invalid_argument("place_idx out of range");
        if (ord.side[j] != 1 && ord.side[j] != -1)
            throw std::invalid_argument("side must be +1 (buy) or -1 (sell)");
    }
}

class Simulator {
  public:
    Simulator(const Messages& msg, const Book& book, const Orders& ord, Model model)
        : msg_(msg), book_(book), ord_(ord), fifo_(model == Model::Fifo),
          touch_(model == Model::Touch), remaining_(ord.size, ord.size + ord.k),
          ahead_(ord.k, 0), behind_(fifo_ ? ord.k : 0) {
        const std::size_t k = ord.k;
        out_.filled_qty.assign(k, 0);
        out_.fill_idx.assign(k, -1);
        out_.fill_time.assign(k, std::numeric_limits<double>::quiet_NaN());
        out_.fill_price.assign(k, 0);
        out_.end_idx.assign(k, -1);
        out_.end_time.assign(k, std::numeric_limits<double>::quiet_NaN());
        out_.end_reason.assign(k, kActive);
        out_.queue_ahead.assign(k, 0);
    }

    Fills run() {
        const std::size_t N = msg_.n, K = ord_.k, L = book_.levels;
        std::vector<std::size_t> order_of(K);
        std::iota(order_of.begin(), order_of.end(), std::size_t{0});
        std::stable_sort(order_of.begin(), order_of.end(), [&](std::size_t a, std::size_t b) {
            return ord_.place_idx[a] < ord_.place_idx[b];
        });
        std::size_t ptr = 0;
        const std::size_t start = K ? static_cast<std::size_t>(ord_.place_idx[order_of[0]]) : N;

        for (std::size_t n = start; n < N; ++n) {
            const double tn = msg_.time[n];
            // (1) timeouts: deadline strictly before this message
            while (!heap_.empty() && heap_.top().first < tn) {
                const auto [dl, j] = heap_.top();
                heap_.pop();
                if (out_.end_reason[j] == kActive)
                    finish(j, kTimeout, static_cast<std::int64_t>(n) - 1, dl);
            }

            // (2) the message itself
            const std::int8_t ty = msg_.type[n];
            if (ty == kTradingHalt) {
                end_all(bids_, kHalt, n, tn);
                end_all(asks_, kHalt, n, tn);
            } else if (!bids_.empty() || !asks_.empty()) {
                const int d = msg_.direction[n];
                const std::int64_t p = msg_.price[n];
                SideBook* own = d == 1 ? &bids_ : (d == -1 ? &asks_ : nullptr);
                const Level* at = nullptr;
                if (own) {
                    auto it = own->find(p);
                    if (it != own->end()) at = &it->second;
                }
                if (ty == kSubmit) {
                    if (fifo_ && at)
                        for (std::size_t j : *at) behind_[j].insert(msg_.order_id[n]);
                } else if (ty == kCancel || ty == kDelete) {
                    if (fifo_ && at) {
                        const std::int64_t oid = msg_.order_id[n], s = msg_.size[n];
                        for (std::size_t j : *at) {
                            if (behind_[j].contains(oid)) {
                                if (ty == kDelete) behind_[j].erase(oid);
                            } else {
                                ahead_[j] = std::max<std::int64_t>(0, ahead_[j] - s);
                            }
                        }
                    }
                } else if (ty == kExecute || ty == kHidden) {
                    // trades at our price (the through model ignores them)
                    if (touch_) {
                        fill_level(bids_, p, n);
                        fill_level(asks_, p, n);
                    } else if (fifo_ && at && ty == kExecute) {
                        const Level copy = *at;
                        for (std::size_t j : copy) {
                            const std::int64_t x = msg_.size[n];
                            const std::int64_t take = std::min(ahead_[j], x);
                            ahead_[j] -= take;
                            if (x - take > 0) fill(j, x - take, n);
                        }
                    }
                    // trades strictly through our price: every model
                    fill_range(bids_, bids_.upper_bound(p), bids_.end(), n);   // buys above p
                    fill_range(asks_, asks_.begin(), asks_.lower_bound(p), n); // sells below p
                }
            }

            // (3) book row n: the opposite quote reaching our price, or our level leaving view
            if (!bids_.empty()) {
                const std::int64_t best_ask = book_.ask_price[n * L];
                const std::int64_t last_bid = book_.bid_price[n * L + L - 1];
                fill_range(bids_, bids_.lower_bound(best_ask), bids_.end(), n);      // >= ask
                end_range(bids_, bids_.begin(), bids_.lower_bound(last_bid), n, tn); // < last bid
            }
            if (!asks_.empty()) {
                const std::int64_t best_bid = book_.bid_price[n * L];
                const std::int64_t last_ask = book_.ask_price[n * L + L - 1];
                fill_range(asks_, asks_.begin(), asks_.upper_bound(best_bid), n);    // <= bid
                end_range(asks_, asks_.upper_bound(last_ask), asks_.end(), n, tn);   // > last ask
            }

            // (4) activate the orders placed after message n
            while (ptr < K && static_cast<std::size_t>(ord_.place_idx[order_of[ptr]]) == n) {
                activate(order_of[ptr++], n, tn);
            }
            if (ptr == K && bids_.empty() && asks_.empty()) break;
        }

        // orders still alive when the data run out
        end_all(bids_, kEndOfData, N - 1, msg_.time[N - 1]);
        end_all(asks_, kEndOfData, N - 1, msg_.time[N - 1]);
        for (std::size_t j = 0; j < K; ++j)
            out_.fill_price[j] = out_.filled_qty[j] > 0 ? ord_.price[j] : 0;
        return std::move(out_);
    }

  private:
    SideBook& side_book(std::size_t j) { return ord_.side[j] > 0 ? bids_ : asks_; }

    void finish(std::size_t j, End why, std::int64_t n, double when) {
        out_.end_reason[j] = why;
        out_.end_idx[j] = n;
        out_.end_time[j] = when;
        SideBook& sb = side_book(j);
        auto it = sb.find(ord_.price[j]);
        Level& lv = it->second;
        lv.erase(std::find(lv.begin(), lv.end(), j));
        if (lv.empty()) sb.erase(it);
        if (fifo_) behind_[j].release();
    }

    void fill(std::size_t j, std::int64_t qty, std::size_t n) {
        qty = std::min(qty, remaining_[j]);
        if (qty <= 0) return;
        remaining_[j] -= qty;
        out_.filled_qty[j] += qty;
        out_.fill_idx[j] = static_cast<std::int64_t>(n);
        out_.fill_time[j] = msg_.time[n];
        if (remaining_[j] == 0) finish(j, kFilled, static_cast<std::int64_t>(n), msg_.time[n]);
    }

    // Fill every order at `price` (iterate over a copy: filling removes orders).
    void fill_level(SideBook& sb, std::int64_t price, std::size_t n) {
        auto it = sb.find(price);
        if (it == sb.end()) return;
        const Level copy = it->second;
        for (std::size_t j : copy) fill(j, remaining_[j], n);
    }

    // Collect the prices of [first, last) before touching the map, then act per price.
    template <class F>
    void for_prices(SideBook::iterator first, SideBook::iterator last, F&& act) {
        if (first == last) return;
        keys_.clear();
        for (; first != last; ++first) keys_.push_back(first->first);
        const std::vector<std::int64_t> prices = keys_;
        for (std::int64_t q : prices) act(q);
    }

    void fill_range(SideBook& sb, SideBook::iterator first, SideBook::iterator last, std::size_t n) {
        for_prices(first, last, [&](std::int64_t q) { fill_level(sb, q, n); });
    }

    void end_range(SideBook& sb, SideBook::iterator first, SideBook::iterator last, std::size_t n,
                   double when) {
        for_prices(first, last, [&](std::int64_t q) {
            auto it = sb.find(q);
            if (it == sb.end()) return;
            const Level copy = it->second;
            for (std::size_t j : copy) finish(j, kLevelExit, static_cast<std::int64_t>(n), when);
        });
    }

    void end_all(SideBook& sb, End why, std::size_t n, double when) {
        std::vector<std::size_t> all;
        for (const auto& [price, lv] : sb) all.insert(all.end(), lv.begin(), lv.end());
        for (std::size_t j : all) finish(j, why, static_cast<std::int64_t>(n), when);
    }

    void activate(std::size_t j, std::size_t n, double tn) {
        const std::size_t L = book_.levels;
        const std::int64_t q = ord_.price[j];
        bool bad;
        if (ord_.side[j] > 0) {
            out_.queue_ahead[j] = depth_at(book_.bid_price, book_.bid_size, L, n, q);
            bad = q >= book_.ask_price[n * L] || q < book_.bid_price[n * L + L - 1];
        } else {
            out_.queue_ahead[j] = depth_at(book_.ask_price, book_.ask_size, L, n, q);
            bad = q <= book_.bid_price[n * L] || q > book_.ask_price[n * L + L - 1];
        }
        if (bad || remaining_[j] <= 0) {
            out_.end_reason[j] = kRejected;
            out_.end_idx[j] = static_cast<std::int64_t>(n);
            out_.end_time[j] = tn;
            return;
        }
        ahead_[j] = out_.queue_ahead[j];
        const double deadline = tn + ord_.max_wait[j];
        side_book(j)[q].push_back(j);
        heap_.emplace(deadline, j);
    }

    const Messages& msg_;
    const Book& book_;
    const Orders& ord_;
    const bool fifo_;
    const bool touch_;
    Fills out_;
    std::vector<std::int64_t> remaining_;
    std::vector<std::int64_t> ahead_;                          // FIFO: shares still ahead of us
    std::vector<IdSet> behind_;                                // FIFO: ids that joined after us
    SideBook bids_, asks_;
    using HeapItem = std::pair<double, std::size_t>;           // (deadline, order)
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap_;
    std::vector<std::int64_t> keys_;                           // scratch buffer for for_prices
};

}  // namespace

Fills simulate(const Messages& msg, const Book& book, const Orders& orders, Model model) {
    validate(msg, book, orders);
    if (orders.k == 0) return Fills{};
    return Simulator(msg, book, orders, model).run();
}

}  // namespace markout
