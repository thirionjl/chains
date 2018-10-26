import abc

import numpy as np

from chains.utils import validate
from .initializers import VarInitializer
from .shape import StaticShape
from .tensor import Tensor
from .utils_broadcasting import remove_broadcasting

__all__ = ["Op", "UnaryOp", "BinaryOp", "Var", "Placeholder",
           "Constant", "ElementWiseBinaryOp", "ElementWiseUnaryOp"]


class Op(abc.ABC):
    def __init__(self):
        self.output = None

    def compute(self):
        pass

    def partials(self, d_output):
        raise RuntimeError(f"{self} does not support derivation")

    def check_incoming_shapes(self, *args):
        pass

    def compute_out_shape(self, *args) -> StaticShape:
        raise NotImplementedError


class UnaryOp(Op, abc.ABC):
    def __init__(self):
        super().__init__()
        self.x = None

    def compute(self, x):
        self.x = x


class BinaryOp(Op, abc.ABC):
    def __init__(self):
        super().__init__()
        self.x = None
        self.y = None

    def compute(self, x, y):
        self.x = x
        self.y = y


class _NoOp(Op):

    def __init__(self, shape: tuple, dtype=np.float32):
        super().__init__()
        validate.is_a("shape", shape, tuple)
        validate.is_float_dtype(dtype)

        self.shape = StaticShape.from_tuple(shape)
        self.dtype = dtype

    def compute_out_shape(self) -> StaticShape:
        return self.shape

    def check(self):
        if self.output is None:
            raise ValueError("A settable cannot be set with None")
        value_shape = StaticShape.of_tensor(self.output)
        if not value_shape.is_assignable_to(self.shape):
            raise ValueError(
                f"{type(self)} accepts values compatible with "
                f"shape {self.shape}, but got {value_shape}")
        if np.dtype(self.dtype) != self.output.dtype:
            raise TypeError(
                f"{type(self)} is configured to accept only dtype {self.dtype},"
                f" but got {self.output.dtype}")


class Var(_NoOp):

    def __init__(self, initializer: VarInitializer, shape: tuple,
                 dtype=np.float32):
        super().__init__(shape, dtype)
        validate.is_a("var_initializer", initializer, VarInitializer)

        if StaticShape.from_tuple(shape).is_unknown():
            raise ValueError(
                "Var should have only known dimensions in declared shape")

        self.initializer = initializer

    def initialize(self):
        self.output = self.initializer.initialize(self.shape.to_numpy(),
                                                  self.dtype)
        self.check()


class Placeholder(_NoOp):
    pass


class Constant(_NoOp):
    def __init__(self, value: Tensor, dtype=np.float32):
        super().__init__(StaticShape.of_tensor(value), dtype)
        self.output = np.array(value).astype(dtype)
        self.check()


class ElementWiseBinaryOp(BinaryOp, abc.ABC):
    def check_incoming_shapes(self, x: StaticShape, y: StaticShape):
        if not x.is_broadcast_compatible(y):
            raise ValueError(
                f"Shapes {x} and {y} cannot be broadcast together")

    def compute_out_shape(self, x: StaticShape, y: StaticShape) -> StaticShape:
        return x.broadcast(y)

    def partials(self, d_output):
        dx, dy = self.simple_derivatives()
        return (
            remove_broadcasting(self.x, d_output * dx),
            remove_broadcasting(self.y, d_output * dy)
        )

    def simple_derivatives(self):
        raise NotImplementedError


class ElementWiseUnaryOp(UnaryOp, abc.ABC):
    def check_incoming_shapes(self, x: StaticShape):
        pass

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        return x_shape

    def partials(self, d_output):
        dx = self.simple_derivative()
        return d_output * dx,

    def simple_derivative(self):
        raise NotImplementedError
