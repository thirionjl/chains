import numpy as np

from .ops_activation import Sigmoid
from .ops import BinaryOp
from .shape import StaticShape
from .tensor import Tensor

__all__ = ["SigmoidCrossEntropyWithLogits", "SigmoidCrossEntropy",
           "SoftMaxCrossEntropyWithLogits", "SoftMaxCrossEntropy"]


class SigmoidCrossEntropyWithLogits(BinaryOp):
    def __init__(self):
        super().__init__()
        self.activations = None

    def compute(self, x: Tensor, y: Tensor):
        super().compute(x, y)
        self.output = np.fmax(x, 0) - x * y + np.log(1 + np.exp(-np.abs(x)))
        self.activations = Sigmoid.sigmoid(self.x)

    def partials(self, d_output):
        return (self.activations - self.y) * d_output, 0  # Do not use dLabels

    def check_incoming_shapes(self, x: StaticShape, y: StaticShape):
        if x != y:
            raise ValueError(
                f"SigmoidCrossEntropy requires operands have same shape, "
                f"got {x} and {y}")

    def compute_out_shape(self, x: StaticShape, y: StaticShape) -> StaticShape:
        return x


class SigmoidCrossEntropy(SigmoidCrossEntropyWithLogits):
    def __init__(self):
        super().__init__()
        self.batch_size = None

    def compute(self, x: Tensor, y: Tensor):
        super().compute(x, y)
        self.batch_size = self.output.size
        self.output = np.mean(self.output)

    def partials(self, d_output):
        d_logits = super().partials(d_output)[0]
        return d_logits / self.batch_size, 0  # Do not use dLabels

    def compute_out_shape(self, x: StaticShape, y: StaticShape) -> StaticShape:
        return StaticShape.scalar()


class SoftMaxCrossEntropyWithLogits(BinaryOp):

    def __init__(self, class_axis=0, keepdims=False):
        super().__init__()
        self.activations = None
        self.class_axis = class_axis
        self.keepdims = keepdims

    def reduce_max(self, x: Tensor):  # dim(x) = (x,y,z,t)
        # dims(max_1) =  (x,y,z,1)
        max_1 = np.max(x, axis=self.class_axis, keepdims=True)
        return np.subtract(x, max_1)  # dim(out) = (x,y,z,t)

    def compute(self, x: Tensor, y: Tensor):  # dim(x) = (x,y,z,t) , dim(y) = t
        super().compute(x, y)

        e = np.exp(self.reduce_max(x))
        s = np.sum(e, axis=self.class_axis, keepdims=True)  # dims (x,y,z,1)
        self.activations = np.divide(e, s)  # dim(a) = (x,y,z,t)

        self.output = -np.sum(y * np.log(self.activations),
                              axis=self.class_axis, keepdims=self.keepdims)

    def partials(self, d_output):
        return (self.activations - self.y) * d_output, 0  # Do not use dLabels

    def check_incoming_shapes(self, x: StaticShape, y: StaticShape):
        if x != y:
            raise ValueError(f"SoftMaxCrossEntropy requires operand have same "
                             f"shape, got {x} and {y}")

    def compute_out_shape(self, x: StaticShape, y: StaticShape) -> StaticShape:
        return StaticShape.from_tuple(x[:-1])


class SoftMaxCrossEntropy(SoftMaxCrossEntropyWithLogits):
    def __init__(self, class_axis=0):
        super().__init__(class_axis=class_axis, keepdims=False)
        self.batch_size = None

    def compute(self, x: Tensor, y: Tensor):
        super().compute(x, y)
        self.batch_size = self.output.size
        self.output = np.mean(self.output)

    def partials(self, d_output):
        d_logits = super().partials(d_output)[0]
        return d_logits / self.batch_size, 0  # Do not use dLabels

    def compute_out_shape(self, x: StaticShape, y: StaticShape) -> StaticShape:
        return StaticShape.scalar()
