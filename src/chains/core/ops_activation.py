"""Collection of activation functions"""
import numpy as np

from .ops import UnaryOp, ElementWiseUnaryOp
from .shape import Shape
from chains.utils.nd_typing import NdArrayLike

__all__ = ["ReLu", "LeakyReLu", "TanH", "Sigmoid", "SoftMax"]


class ReLu(ElementWiseUnaryOp):
    """Implements the rectified linear unit operation"""

    def compute(self, x: NdArrayLike):
        self.output = np.fmax(x, 0)

    def simple_derivative(self):
        return np.sign(self.output)


class LeakyReLu(ElementWiseUnaryOp):
    """Implements the leaky rectified linear unit operation"""

    def __init__(self, leak: float = 0.01):
        """Creates the op
        :param leak: A float representing the leak (Derivative on negatives)"""
        super().__init__()
        self.leak = leak

    def compute(self, x: NdArrayLike):
        super().compute(x)
        self.output = np.fmax(self.leak * x, x)

    def simple_derivative(self):
        return np.fmax(np.sign(self.x), self.leak)

    def compute_out_dtype(self, dtype):
        return np.result_type(dtype, self.leak)


class TanH(ElementWiseUnaryOp):
    """Hyperbolic tangent activation function"""

    def compute(self, x: NdArrayLike):
        self.output = np.tanh(x)

    def simple_derivative(self):
        sq = self.output**2
        return 1 - sq


class Sigmoid(ElementWiseUnaryOp):
    """Sigmoid activation function. Used for binary classification"""

    @staticmethod
    def sigmoid(x: NdArrayLike):
        return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

    def compute(self, x: NdArrayLike):
        self.output = Sigmoid.sigmoid(x)

    def simple_derivative(self):
        return self.output * (1 - self.output)


class SoftMax(UnaryOp):
    """Softmax activation function"""

    def __init__(self, class_axis=0):
        """:param class_axis: The axis of the input on which to find the values
        for the different classes
        """
        super().__init__()
        self.class_axis = class_axis

    @staticmethod
    def reduce_max(x: NdArrayLike, class_axis):
        # dim(x) =  dim(out) = (x,y,z,t)
        # dims(max_1) =  (x,y,z,1)
        max_1 = np.max(x, axis=class_axis, keepdims=True)
        return np.subtract(x, max_1)

    @staticmethod
    def softmax(x: NdArrayLike, class_axis):
        # dim(x) =  dim(e) =  dim(out) = (x,y,z,t)
        # dim(s) = (x,y,z,1)
        e = np.exp(SoftMax.reduce_max(x, class_axis))
        s = np.sum(e, axis=class_axis, keepdims=True)
        return np.divide(e, s)

    def compute(self, x: NdArrayLike):
        super().compute(x)
        self.output = SoftMax.softmax(x, self.class_axis)

    def check_incoming_shapes(self, x: Shape):
        pass

    def compute_out_shape(self, x_shape: Shape) -> Shape:
        return x_shape

    def partials(self, d_output):
        raise NotImplementedError("todo")
