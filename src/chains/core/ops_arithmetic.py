"""Common arithmetic operations"""
from functools import reduce

import numpy as np

from chains.utils.nd_typing import NdArrayLike
from .ops import ElementWiseBinaryOp, ElementWiseUnaryOp, Op
from .shape import Shape
from ..utils import validate

__all__ = ["Add", "Negate", "ConstMul", "Mul", "Pow", "IsGreaterThan"]


class Add(ElementWiseBinaryOp):
    def compute(self, x, y):
        self.output = x + y

    def simple_derivatives(self):
        return 1, 1


class AddScalars(Op):
    def __init__(self):
        super().__init__()
        self.cnt_incoming_args = None

    def compute(self, *args: NdArrayLike):
        self.output = reduce(lambda x, y: x.item() + y.item(), args, 0)

    def partials(self, d_output: NdArrayLike):
        return tuple(1 for _ in range(self.cnt_incoming_args))

    def check_incoming_shapes(self, *static_shapes: Shape):
        self.cnt_incoming_args = len(static_shapes)
        if any(s.size() > 1 for s in static_shapes):
            raise ValueError(f"All incoming shapes must contain exactly one element")

    def compute_out_shape(self, *static_shapes: Shape) -> Shape:
        return Shape.scalar()

    def compute_out_dtype(self, *dtypes: np.dtype):
        return super().compute_out_dtype(*dtypes)


class Negate(ElementWiseUnaryOp):
    def compute(self, x: NdArrayLike):
        self.output = -x

    def simple_derivative(self):
        return -1


class ConstMul(ElementWiseUnaryOp):
    def __init__(self, c: NdArrayLike):
        super().__init__()
        self.c = c

    def compute(self, x: NdArrayLike):
        self.output = self.c * x

    def simple_derivative(self):
        return self.c

    def compute_out_dtype(self, dtype):
        return np.result_type(self.c, dtype)


class Mul(ElementWiseBinaryOp):
    def compute(self, x: NdArrayLike, y: NdArrayLike):
        super().compute(x, y)
        self.output = x * y

    def simple_derivatives(self):
        return self.y, self.x


class Pow(ElementWiseUnaryOp):
    """Exponentiation by an integer"""

    def __init__(self, exponent: int):
        super().__init__()
        validate.is_integer_dtype(int)
        self.exponent = exponent

    def compute(self, x: NdArrayLike):
        super().compute(x)
        self.output = x**self.exponent

    def simple_derivative(self):
        return self.exponent * self.x ** (self.exponent - 1)

    def compute_out_dtype(self, dtype):
        return np.result_type(self.exponent, dtype)


class IsGreaterThan(ElementWiseUnaryOp):
    """Returns a boolean indicating if the input is greater that some output.
    Note: Input derivatives are not computed.
    """

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def compute(self, x):
        self.output = (x > self.threshold).astype(np.int8)

    def compute_out_dtype(self, dtype):
        return np.int8

    def simple_derivative(self):
        return 0
