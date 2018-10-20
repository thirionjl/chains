import numpy as np

from .ops import ElementWiseUnaryOp
from .tensor import Tensor

__all__ = ["ReLu", "LeakyReLu", "TanH", "Sigmoid"]


class ReLu(ElementWiseUnaryOp):

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = np.fmax(x, 0)

    def simple_derivative(self):
        return np.sign(self.output)


class LeakyReLu(ElementWiseUnaryOp):

    def __init__(self, leak: float = 0.001):
        super().__init__()
        self.leak = leak

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = np.fmax(self.leak * x, x)

    def simple_derivative(self):
        return np.fmax(np.sign(self.x), self.leak)


class TanH(ElementWiseUnaryOp):

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = np.tanh(x)

    def simple_derivative(self):
        sq = self.output ** 2
        return 1 - sq


class Sigmoid(ElementWiseUnaryOp):

    @staticmethod
    def sigmoid(x: Tensor):
        return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = self.sigmoid(x)

    def simple_derivative(self):
        return self.output * (1 - self.output)
