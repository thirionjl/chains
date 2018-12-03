"""Common arithmetic operations"""
import numpy as np

from chains.utils import validate
from .ops import ElementWiseBinaryOp, ElementWiseUnaryOp
from .tensor import Tensor

__all__ = ["Add", "Negate", "ConstMul", "Mul", "Pow", "IsGreaterThan"]


class Add(ElementWiseBinaryOp):

    def compute(self, x, y):
        super().compute(x, y)
        self.output = x + y

    def simple_derivatives(self):
        return 1, 1


class Negate(ElementWiseUnaryOp):
    def compute(self, x: Tensor):
        super().compute(x)
        self.output = -x

    def simple_derivative(self):
        return -1


class ConstMul(ElementWiseUnaryOp):
    def __init__(self, c: Tensor):
        super().__init__()
        self.c = c

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = self.c * x

    def simple_derivative(self):
        return self.c

    def compute_out_dtype(self, dtype):
        return np.result_type(self.c, dtype)


class Mul(ElementWiseBinaryOp):

    def compute(self, x: Tensor, y: Tensor):
        super().compute(x, y)
        self.output = x * y

    def simple_derivatives(self):
        return self.y, self.x


class Pow(ElementWiseUnaryOp):
    """Exponentiation by an integer"""

    def __init__(self, exponent: int):
        validate.is_integer_dtype(int)
        self.exponent = exponent

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = x ** self.exponent

    def simple_derivative(self):
        return self.exponent * self.x ** (self.exponent - 1)

    def compute_out_dtype(self, dtype):
        return np.result_type(self.exponent, dtype)


class IsGreaterThan(ElementWiseUnaryOp):
    """Returns a boolean indicating if the input is greater that some output.
    Note: Input derivatives are not computed.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def compute(self, x):
        super().compute(x)
        self.output = (x > self.threshold).astype(np.int8)

    def compute_out_dtype(self, dtype):
        return np.int8

    def simple_derivative(self):
        return 0
