"""Type stub for the pybind11 extension built from cpp/bindings.cpp."""

import numpy as np
from numpy.typing import ArrayLike

MODELS: tuple[str, str, str]

def simulate(
    time: ArrayLike,
    type: ArrayLike,
    order_id: ArrayLike,
    size: ArrayLike,
    price: ArrayLike,
    direction: ArrayLike,
    bid_price: ArrayLike,
    bid_size: ArrayLike,
    ask_price: ArrayLike,
    ask_size: ArrayLike,
    place_idx: ArrayLike,
    side: ArrayLike,
    order_price: ArrayLike,
    order_size: ArrayLike,
    max_wait: ArrayLike,
    model: int,
) -> dict[str, np.ndarray]: ...
