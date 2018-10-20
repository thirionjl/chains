import numpy as np

from chains.core.ops import UnaryOp
from chains.core.shape import StaticShape
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


class SoftMax(UnaryOp):

    def __init__(self, class_axis=0):
        super().__init__()
        self.class_axis = class_axis

    @staticmethod
    def reduce_max(x: Tensor, class_axis):
        # dim(x) =  dim(out) = (x,y,z,t)
        # dims(max_1) =  (x,y,z,1)
        max_1 = np.max(x, axis=class_axis, keepdims=True)
        return np.subtract(x, max_1)

    @staticmethod
    def softmax(x: Tensor, class_axis):
        # dim(x) =  dim(e) =  dim(out) = (x,y,z,t)
        # dim(s) = (x,y,z,1)
        e = np.exp(SoftMax.reduce_max(x, class_axis))
        s = np.sum(e, axis=class_axis, keepdims=True)
        return np.divide(e, s)

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = SoftMax.softmax(x, self.class_axis)

    def check_incoming_shapes(self, x: StaticShape):
        pass

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        return x_shape

    def partials(self, d_output):
        raise NotImplementedError("todo")
