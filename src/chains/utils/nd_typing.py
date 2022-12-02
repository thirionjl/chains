from typing import Any, TypeGuard

import numpy as np
from numpy import typing

__all__ = ["NdArrayLike", "is_ndarray_like", "NdShape"]

NdArrayLike = typing.NDArray | int | float
NdArray = typing.NDArray
NdShape = tuple[int, ...]


def is_ndarray_like(x: Any) -> TypeGuard[NdArrayLike]:
    try:
        return np.can_cast(np.array(x), np.float64)
    except (ValueError, TypeError):
        return False
