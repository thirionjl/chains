import abc

import numpy as np

from .shape import StaticShape

__all__ = ["VarInitializer", "ConstantInitializer", "ZeroInitializer",
           "RandomNormalInitializer", "HeInitializer", "XavierInitializer"]


class VarInitializer(abc.ABC):

    def initialize(self, param, dtype):
        raise NotImplementedError


class ConstantInitializer(VarInitializer):
    def __init__(self, value):
        self.value = value
        self.shape = np.shape(value)
        self.dtype = np.array(value).dtype

    def initialize(self, shape, dtype):
        if self.shape != shape:
            raise ValueError(f"Constant initialization does not match "
                             f"required shape {shape}")
        if self.dtype != dtype:
            raise ValueError(f"Constant initialization does not match "
                             f"required dtype {shape}")
        return self.value


class ZeroInitializer(VarInitializer):
    def initialize(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)


class RandomNormalInitializer(VarInitializer):

    def __init__(self, scale: float = 0.01):
        self.scale = scale

    def initialize(self, shape, dtype):
        if dtype != np.float32:
            raise ValueError(f"Random initialization does produce only "
                             f"np.float32 but got var with dtype={dtype}")
        return np.random.randn(*shape).astype(np.float32) * self.scale


class KeepVarianceInitializer(RandomNormalInitializer, abc.ABC):

    def __init__(self, k=1, axis_size_divider=-1):
        super().__init__(scale=1)
        self.k = k
        self.axis_size_divider = axis_size_divider

    def initialize(self, shape, dtype):
        StaticShape.from_tuple(shape).check_axis_index(self.axis_size_divider)
        r = super().initialize(shape, dtype)
        sq = np.sqrt(self.k / shape[self.axis_size_divider], dtype=np.float32)
        return r * sq


class HeInitializer(KeepVarianceInitializer):
    def __init__(self, axis_size_divider=-1):
        super().__init__(k=2, axis_size_divider=axis_size_divider)


class XavierInitializer(KeepVarianceInitializer):
    def __init__(self, axis_size_divider=-1):
        super().__init__(k=1, axis_size_divider=axis_size_divider)
