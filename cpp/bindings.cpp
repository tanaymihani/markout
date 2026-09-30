// pybind11 bindings: markout.lob._queue_sim.simulate(...) on NumPy arrays.
//
// Arrays are taken as C-contiguous views of the dtypes the Python side already uses
// (float64 times, int8 type/direction/side, int64 everything else), so no copies are
// made for the data produced by markout.lob.lobster. Output vectors are moved into
// NumPy arrays that own them (no copy on the way out either).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "queue_sim.hpp"

namespace py = pybind11;

namespace {

template <class T>
using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <class T>
const T* vector_data(const Array<T>& a, py::ssize_t n, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != n)
        throw std::invalid_argument(std::string(name) + ": expected a 1-D array of length " +
                                    std::to_string(n));
    return a.data();
}

template <class T>
const T* matrix_data(const Array<T>& a, py::ssize_t n, py::ssize_t levels, const char* name) {
    if (a.ndim() != 2 || a.shape(0) != n || a.shape(1) != levels)
        throw std::invalid_argument(std::string(name) + ": expected shape (n, levels)");
    return a.data();
}

template <class T>
py::array_t<T> to_numpy(std::vector<T>&& v) {
    auto* owned = new std::vector<T>(std::move(v));
    py::capsule free_when_done(owned, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    return py::array_t<T>(static_cast<py::ssize_t>(owned->size()), owned->data(), free_when_done);
}

py::dict simulate(const Array<double>& time, const Array<std::int8_t>& type,
                  const Array<std::int64_t>& order_id, const Array<std::int64_t>& size,
                  const Array<std::int64_t>& price, const Array<std::int8_t>& direction,
                  const Array<std::int64_t>& bid_price, const Array<std::int64_t>& bid_size,
                  const Array<std::int64_t>& ask_price, const Array<std::int64_t>& ask_size,
                  const Array<std::int64_t>& place_idx, const Array<std::int8_t>& side,
                  const Array<std::int64_t>& order_price, const Array<std::int64_t>& order_size,
                  const Array<double>& max_wait, int model) {
    if (model < 0 || model > 2) throw std::invalid_argument("model must be 0 (touch), 1 (fifo) or 2 (through)");
    if (time.ndim() != 1) throw std::invalid_argument("time: expected a 1-D array");
    if (bid_price.ndim() != 2) throw std::invalid_argument("bid_price: expected a 2-D array");
    const py::ssize_t n = time.shape(0), levels = bid_price.shape(1), k = place_idx.shape(0);

    const markout::Messages msg{vector_data(time, n, "time"), vector_data(type, n, "type"),
                                vector_data(order_id, n, "order_id"), vector_data(size, n, "size"),
                                vector_data(price, n, "price"),
                                vector_data(direction, n, "direction"), static_cast<std::size_t>(n)};
    const markout::Book book{matrix_data(bid_price, n, levels, "bid_price"),
                             matrix_data(bid_size, n, levels, "bid_size"),
                             matrix_data(ask_price, n, levels, "ask_price"),
                             matrix_data(ask_size, n, levels, "ask_size"),
                             static_cast<std::size_t>(n), static_cast<std::size_t>(levels)};
    const markout::Orders orders{vector_data(place_idx, k, "place_idx"), vector_data(side, k, "side"),
                                 vector_data(order_price, k, "order price"),
                                 vector_data(order_size, k, "order size"),
                                 vector_data(max_wait, k, "max_wait"), static_cast<std::size_t>(k)};
    markout::Fills f;
    {
        py::gil_scoped_release release;   // the arrays above stay alive for this scope
        f = markout::simulate(msg, book, orders, static_cast<markout::Model>(model));
    }
    py::dict out;
    out["filled_qty"] = to_numpy(std::move(f.filled_qty));
    out["fill_idx"] = to_numpy(std::move(f.fill_idx));
    out["fill_time"] = to_numpy(std::move(f.fill_time));
    out["fill_price"] = to_numpy(std::move(f.fill_price));
    out["end_idx"] = to_numpy(std::move(f.end_idx));
    out["end_time"] = to_numpy(std::move(f.end_time));
    out["end_reason"] = to_numpy(std::move(f.end_reason));
    out["queue_ahead"] = to_numpy(std::move(f.queue_ahead));
    return out;
}

}  // namespace

PYBIND11_MODULE(_queue_sim, m) {
    m.doc() = "C++17 port of markout.lob.fills.simulate_python (bit-identical output).";
    m.def("simulate", &simulate, py::arg("time"), py::arg("type"), py::arg("order_id"),
          py::arg("size"), py::arg("price"), py::arg("direction"), py::arg("bid_price"),
          py::arg("bid_size"), py::arg("ask_price"), py::arg("ask_size"), py::arg("place_idx"),
          py::arg("side"), py::arg("order_price"), py::arg("order_size"), py::arg("max_wait"),
          py::arg("model"),
          "Replay hypothetical orders against a LOBSTER day under one fill model "
          "(0 touch, 1 fifo, 2 through). Returns a dict of NumPy arrays, one entry per order.");
    m.attr("MODELS") = py::make_tuple("touch", "fifo", "through");
}
