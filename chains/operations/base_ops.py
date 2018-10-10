import abc

import numpy as np

import chains.initialization.variable_initializers as init
import chains.operations.broadcasting_utils as utils
from chains.tensor.tensor import Tensor, Shape


class Op(abc.ABC):
    def __init__(self):
        self.output = None

    def compute(self):
        pass

    def partials(self, d_output):
        raise RuntimeError(f"{self} does not support derivation")

    def check_incoming_shapes(self, *args):
        pass

    def compute_out_shape(self, *args) -> Shape:
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


class NoOp(Op):

    def __init__(self, shape: Shape, dtype=np.float64):
        super().__init__()
        if not isinstance(shape, Shape):
            raise ValueError("Did not get a shape object")
        self.shape = shape
        self.dtype = dtype

    def compute_out_shape(self) -> Shape:
        return self.shape

    def check(self):
        if self.output is None:
            raise ValueError("A settable cannot be set with None")
        value_shape = Shape.of_tensor(self.output)
        if not value_shape.is_assignable_to(self.shape):
            raise ValueError(
                f"{type(self)} accepts values compatible with "
                f"shape {self.shape}, but got {value_shape}")


class Var(NoOp):

    def __init__(self, initializer, shape, dtype=np.float64):
        if not isinstance(initializer, init.VarInitializer):
            raise ValueError("Var should be passed a VarInitializer subclass")
        if not isinstance(shape, Shape):
            raise ValueError("Did not get a shape object")
        if shape.is_unknown():
            raise ValueError(
                "Var should have only known dimensions in declared shape")

        self.initializer = initializer
        super().__init__(shape, dtype)

    def initialize(self):
        self.output = self.initializer.initialize(self.shape.to_numpy(),
                                                  self.dtype)
        self.check()


class Placeholder(NoOp):
    pass


class Constant(NoOp):
    def __init__(self, value: Tensor):
        super().__init__(Shape.of_tensor(value), np.array(value).dtype)
        self.output = value
        self.check()


class ElementWiseBinaryOp(BinaryOp, abc.ABC):
    def check_incoming_shapes(self, x: Shape, y: Shape):
        if not x.is_broadcast_compatible(y):
            raise ValueError(
                f"Shapes {x} and {y} cannot be broadcast together")

    def compute_out_shape(self, x: Shape, y: Shape) -> Shape:
        return x.broadcast(y)

    def partials(self, d_output):
        dx, dy = self.simple_derivatives()
        return (
            utils.remove_broadcasting(self.x, d_output * dx),
            utils.remove_broadcasting(self.y, d_output * dy)
        )

    def simple_derivatives(self):
        raise NotImplementedError


class ElementWiseUnaryOp(UnaryOp, abc.ABC):
    def check_incoming_shapes(self, x: Shape):
        pass

    def compute_out_shape(self, x_shape: Shape) -> Shape:
        return x_shape

    def partials(self, d_output):
        dx = self.simple_derivative()
        return d_output * dx,

    def simple_derivative(self):
        raise NotImplementedError
