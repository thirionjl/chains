import abc

import numpy as np

from chains.tensor.tensor import Shape


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
            raise ValueError(f"Constant initialization does not match required shape {shape}")
        if self.dtype != dtype:
            raise ValueError(f"Constant initialization does not match required dtype {shape}")
        return self.value


class ZeroInitializer(VarInitializer):
    def initialize(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)


class RandomNormalInitializer(VarInitializer):

    def __init__(self, scale: float = 0.01, seed=None):
        self.scale = scale
        np.random.seed(seed)

    def initialize(self, shape, dtype):
        if dtype != np.float64:
            raise ValueError(f"Random initialization does produce only np.float64 but got var with dtype={dtype}")
        return np.random.randn(*shape) * self.scale


class KeepVarianceInitializer(RandomNormalInitializer, abc.ABC):

    def __init__(self, seed=None, k=1, axis_size_divider=-1):
        super().__init__(scale=1, seed=seed)
        self.k = k
        self.axis_size_divider = axis_size_divider

    def initialize(self, shape, dtype):
        Shape.from_tuple(shape).check_axis_index(self.axis_size_divider)
        r = super().initialize(shape, dtype)
        return r * np.sqrt(self.k / shape[self.axis_size_divider])


class HeInitializer(KeepVarianceInitializer):
    def __init__(self, seed=None, axis_size_divider=-1):
        super().__init__(seed=seed, k=2, axis_size_divider=axis_size_divider)


class XavierInitializer(KeepVarianceInitializer):
    def __init__(self, seed=None, axis_size_divider=-1):
        super().__init__(seed=seed, k=1, axis_size_divider=axis_size_divider)
