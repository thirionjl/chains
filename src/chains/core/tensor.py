from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy import ndarray

__all__ = ["Tensor", "is_tensor", "Shape"]

Tensor = ndarray | Iterable | int | float
Shape = tuple[int, ...]


def is_tensor(x: Any):
    try:
        np.reshape()
        return np.can_cast(np.array(x), np.float64)
    except (ValueError, TypeError):
        return False
