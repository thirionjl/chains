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
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = self.c * x

    def simple_derivative(self):
        return self.c


class Mul(ElementWiseBinaryOp):

    def compute(self, x: Tensor, y: Tensor):
        super().compute(x, y)
        self.output = x * y

    def simple_derivatives(self):
        return self.y, self.x


class Pow(ElementWiseUnaryOp):

    def __init__(self, exponent: int):
        self.exponent = exponent

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = x ** self.exponent

    def simple_derivative(self):
        return self.exponent * self.x ** (self.exponent - 1)


class IsGreaterThan(ElementWiseUnaryOp):

    def __init__(self, threshold: float):
        self.threshold = threshold

    def compute(self, x):
        super().compute(x)
        self.output = (x > self.threshold).astype(int)

    def simple_derivative(self):
        return 0
