// Fill simulator for hypothetical passive orders replayed against a LOBSTER day.
//
// A C++17 port of markout.lob.fills.simulate_python, the Python reference. The algorithm
// and its order of operations are identical, so the two must agree bit for bit
// (tests/test_cpp_*.py checks this on all five stocks and on random event streams).
//
// For message n: (1) expire orders whose deadline is before time[n]; (2) apply the
// message (FIFO queue bookkeeping, fills at our price, prints through our price);
// (3) check book row n for the opposite quote reaching our price and for our price
// leaving the visible levels; (4) activate orders placed after message n. Every
// order's outcome depends only on its own parameters and the message stream, never on
// the other hypothetical orders, so the order in which concurrent orders are visited
// within a message cannot change any result.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace markout {

enum class Model : int { Touch = 0, Fifo = 1, Through = 2 };

enum End : std::int8_t {
    kActive = 0,
    kFilled = 1,
    kTimeout = 2,
    kLevelExit = 3,
    kHalt = 4,
    kEndOfData = 5,
    kRejected = 6,
};

// LOBSTER message types.
enum MsgType : std::int8_t {
    kSubmit = 1,
    kCancel = 2,
    kDelete = 3,
    kExecute = 4,
    kHidden = 5,
    kCross = 6,
    kTradingHalt = 7,
};

// Non-owning views of the NumPy arrays (row-major book blocks of shape n x levels).
struct Messages {
    const double* time;
    const std::int8_t* type;
    const std::int64_t* order_id;
    const std::int64_t* size;
    const std::int64_t* price;
    const std::int8_t* direction;
    std::size_t n;
};

struct Book {
    const std::int64_t* bid_price;
    const std::int64_t* bid_size;
    const std::int64_t* ask_price;
    const std::int64_t* ask_size;
    std::size_t n;
    std::size_t levels;
};

struct Orders {
    const std::int64_t* place_idx;
    const std::int8_t* side;       // +1 buy, -1 sell
    const std::int64_t* price;
    const std::int64_t* size;
    const double* max_wait;        // seconds after time[place_idx]
    std::size_t k;
};

struct Fills {
    std::vector<std::int64_t> filled_qty;
    std::vector<std::int64_t> fill_idx;     // -1 if nothing filled
    std::vector<double> fill_time;          // NaN if nothing filled
    std::vector<std::int64_t> fill_price;   // limit price if anything filled, else 0
    std::vector<std::int64_t> end_idx;
    std::vector<double> end_time;
    std::vector<std::int8_t> end_reason;
    std::vector<std::int64_t> queue_ahead;  // displayed depth ahead of us at placement
};

// Throws std::invalid_argument on inconsistent inputs.
Fills simulate(const Messages& msg, const Book& book, const Orders& orders, Model model);

}  // namespace markout
