from typing import Union, Iterable

import numpy as np

__all__ = ["Tensor", "is_tensor"]

Tensor = Union[int, float, np.ndarray, Iterable]


def is_tensor(x):
    return isinstance(x, Tensor.__args__)
