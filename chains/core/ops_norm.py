import numpy as np

from chains.core.ops import Op
from chains.core.shape import Dim
from chains.utils import validate
from .shape import StaticShape
from .tensor import Tensor

__all__ = ["BatchNorm"]


class BatchNorm(Op):

    def __init__(self, epsilon=1e-3, sample_axis=-1):
        validate.is_one_of("sample_axis", sample_axis, (0, -1))
        self.sample_axis = sample_axis
        self.epsilon = epsilon
        self.beta = None
        self.gamma = None
        self.mu = None
        self.s2 = None
        self.x_hat = None
        self.sq = None

    def check_incoming_shapes(self, beta: StaticShape, gamma: StaticShape,
                              x: StaticShape):
        if len(x) != 2:
            raise ValueError(f"BatchNorm requires a 2-D matrix as input")

            features_dim = x[self.sample_axis + 1]

        _verify_vector("beta", beta, features_dim)
        _verify_vector("gamma", gamma, features_dim)

    def _verify_vector(self, name: str, shape: StaticShape, features_dim: Dim):
        correct_shape = []
        correct_shape[self.sample_axis] = Dim.of(1)
        correct_shape[self.sample_axis + 1] = features_dim
        StaticShape.from_tuple(correct_shape)
        if not correct_shape == shape:
            raise ValueError(f"Incorrect shape for {name} should be "
                             f"{correct_shape}, but got {shape}")

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        return x_shape

    def compute(self, beta: Tensor, gamma: Tensor, x: Tensor):
        super().compute(x)
        self.beta = beta
        self.gamma = gamma
        mu = np.mean(x, axis=self.sample_axis)
        s2 = np.var(x, axis=self.sample_axis)
        self.sq = np.sqrt(s2 + self.epsilon)
        self.x_hat = (x - mu) / self.sq
        self.output = gamma * self.x_hat + beta

    def partials(self, d_out):
        x_hat = self.x_hat
        dx_hat = self.gamma * d_out
        axis = self.sample_axis

        m = np.shape(d_out, axis=axis)
        p1 = m * dx_hat
        p2 = -np.sum(dx_hat, axis=axis)
        p3 = - x_hat * np.sum(x_hat * dx_hat, axis=axis)
        d_x = (p1 + p2 + p3) / (m * self.sq)

        d_beta = np.sum(d_out, axis=axis)
        d_gamma = np.sum(x_hat * d_out, axis=axis)

        return d_beta, d_gamma, d_x
