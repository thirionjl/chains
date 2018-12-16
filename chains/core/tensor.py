from typing import Union, Iterable, Tuple

import numpy as np

__all__ = ["Tensor", "is_tensor", "Shape"]

Tensor = Union[int, float, np.ndarray, Iterable]
Shape = Tuple[int]


def is_tensor(x):
    return isinstance(x, Tensor.__args__)
