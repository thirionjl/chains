import numpy as np

from .ops import Op, ElementWiseUnaryOp
from .static_shape import StaticShape
from .tensor import Tensor

__all__ = ["L2NormRegularization", "Dropout"]


class L2NormRegularization(Op):
    def __init__(self, lambd: float, epsilon=1e-12):
        super().__init__()
        if lambd is None:
            raise ValueError("L2NormRegularization parameter is mandatory")
        if not isinstance(lambd, float):
            raise ValueError(
                f"L2NormRegularization parameter should be a float, "
                f"got {type(lambd)}"
            )

        self.lambd = lambd
        self.epsilon = epsilon
        self.weights = None
        self.factor = None
        self.batch_size = None

    def check_incoming_shapes(self, batch_size: StaticShape, *weight_shapes):
        if not batch_size.is_scalar:
            raise ValueError("First argument must be scalar")

    def compute_out_shape(self, *static_shapes) -> StaticShape:
        return StaticShape.scalar()

    def compute_out_dtype(self, *dtypes):
        all_dtypes = dtypes[1:] + (self.lambd, self.epsilon)
        return np.result_type(*all_dtypes)

    def compute(self, batch_size, *weights: tuple):
        self.batch_size = batch_size
        self.weights = weights
        self.factor = self.lambd / self.batch_size
        s = 0
        for x in weights:
            xr = np.ravel(x)
            s += np.dot(xr, xr)

        self.output = max(s, self.epsilon) * (self.factor / 2)

    def partials(self, d_output):
        return [-self.output * d_output / self.batch_size] + [
            self.factor * w * d_output for w in self.weights
        ]


class Dropout(ElementWiseUnaryOp):
    def __init__(self, keep_prob):
        if type(keep_prob) != float or not (0 < keep_prob <= 1):
            raise ValueError(
                f"Dropout keep probability should be a float "
                f"between 0 and 1, got {keep_prob}"
            )
        self.keep_prob = keep_prob
        self.mask = None

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        return x_shape

    def compute_out_dtype(self, dtype):
        return np.result_type(dtype, self.keep_prob, np.bool)

    def compute(self, x: Tensor):
        super().compute(x)
        self.mask = np.random.random_sample(np.shape(x)) < self.keep_prob
        self.output = self.mask * self.x / self.keep_prob

    def simple_derivative(self):
        return self.mask / self.keep_prob
