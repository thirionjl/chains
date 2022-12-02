"""
VarInitializers are called by Var Ops for initialization when they are asked
to do so.

Given a dtype and a concrete shape (a tuple of ints - no unknown dimension)
a VarInitialize should produce a Tensor that conforms to that dtype and shape
"""
import abc

import numpy as np

from .shape import Shape
from ..utils import validate

__all__ = [
    "VarInitializer",
    "ConstantInitializer",
    "ZeroInitializer",
    "RandomNormalInitializer",
    "HeInitializer",
    "XavierInitializer",
]


class VarInitializer(abc.ABC):
    """Abstract base class for variable initializers"""

    def initialize(self, shape, dtype):
        """
        :param shape: A tuple of int. (Not a StaticShape)
        :param dtype: A numpy dtype
        :return: ATensor with the shape and dtype provided as arguments
        """
        raise NotImplementedError


class ConstantInitializer(VarInitializer):
    """Initializes from a fixed value. If Var that uses it does not conform
    to the shape and dtype of this fixed value an exception is thrown at
    runtime"""

    def __init__(self, value):
        self.value = value
        self.shape = np.shape(value)
        self.dtype = np.array(value).dtype

    def initialize(self, shape, dtype):
        if self.shape != shape:
            raise ValueError(
                f"Constant initialization does not match " f"required shape {shape}"
            )
        if self.dtype != dtype:
            raise ValueError(
                f"Constant initialization does not match " f"required dtype {shape}"
            )
        return self.value


class ZeroInitializer(VarInitializer):
    """Initializes Tensor with all zeros"""

    def initialize(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)


class OneInitializer(VarInitializer):
    """Initializes Tensor with all ones"""

    def initialize(self, shape, dtype):
        return np.ones(shape, dtype=dtype)


class RandomNormalInitializer(VarInitializer):
    """Initializes from a Random Normal distribution in [0, scale[ interval"""

    def __init__(self, scale: float = 0.01):
        """:param scale: Random-Normal outputs will range from 0 to scale"""
        self.scale = scale

    def initialize(self, shape, dtype):
        validate.is_number_dtype(dtype)
        return np.random.randn(*shape).astype(dtype, copy=False) * self.scale


class KeepVarianceInitializer(RandomNormalInitializer, abc.ABC):
    def __init__(self, k=1, axis_size_divider=-1):
        super().__init__(scale=1)
        self.k = k
        self.axis_size_divider = axis_size_divider

    def initialize(self, shape, dtype):
        Shape.from_tuple(shape).check_axis_index(self.axis_size_divider)
        r = super().initialize(shape, dtype)
        sq = np.sqrt(self.k / shape[self.axis_size_divider], dtype=dtype)
        return r * sq


class HeInitializer(KeepVarianceInitializer):
    """Applies He &  Al. Initialization
    :param axis_size_divider: Axis index used to count the number of samples
    """

    def __init__(self, axis_size_divider=-1):
        super().__init__(k=2, axis_size_divider=axis_size_divider)


class XavierInitializer(KeepVarianceInitializer):
    """Applies Xavier Initialization
    :param axis_size_divider: Axis index used to count the number of samples
    """

    def __init__(self, axis_size_divider=-1):
        super().__init__(k=1, axis_size_divider=axis_size_divider)
