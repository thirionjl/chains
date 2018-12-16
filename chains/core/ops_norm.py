import numpy as np

from .ops import Op
from .static_shape import Dim, StaticShape
from .tensor import Tensor
from ..utils import validate

__all__ = ["BatchNormTraining", "BatchNormPredict"]


class BatchNormTraining(Op):

    def __init__(self, momentum=0.99, epsilon=1e-3, sample_axis=-1):
        validate.is_one_of("sample_axis", sample_axis, (0, -1))
        self.sample_axis = sample_axis
        self.epsilon = epsilon
        self.momentum = momentum
        self.beta = None
        self.gamma = None
        self.s2 = None
        self.x_hat = None
        self.sq = None
        self.avg = 0
        self.var = 0

    def check_incoming_shapes(self, beta: StaticShape, gamma: StaticShape,
                              x: StaticShape):
        self._verify_shapes(self.sample_axis, beta, gamma, x)

    @staticmethod
    def _verify_shapes(axis, beta, gamma, x):
        assert axis in (-1, 0)
        if len(x) != 2:
            raise ValueError(f"BatchNorm requires a 2-D matrix as input")
        features_dim = x[axis + 1]
        BatchNormTraining._verify_vector(axis, "beta", beta, features_dim)
        BatchNormTraining._verify_vector(axis, "gamma", gamma, features_dim)

    @staticmethod
    def _verify_vector(axis, name: str, shape: StaticShape, features_dim: Dim):
        assert axis in (-1, 0)
        s = [Dim.unknown(), Dim.unknown()]
        s[axis] = Dim.of(1)
        s[axis + 1] = features_dim
        correct_shape = StaticShape.from_tuple(s)
        if not correct_shape == shape:
            raise ValueError(f"Incorrect shape for {name} should be "
                             f"{correct_shape}, but got {shape}")

    def compute_out_shape(self, beta_shape, gamma_shape,
                          x_shape: StaticShape) -> StaticShape:
        return x_shape

    def compute(self, beta: Tensor, gamma: Tensor, x: Tensor):
        self.beta = beta
        self.gamma = gamma
        mu = np.mean(x, axis=self.sample_axis, keepdims=True)
        var = np.var(x, axis=self.sample_axis, keepdims=True)
        self.sq = np.sqrt(var + self.epsilon)
        self.x_hat = (x - mu) / self.sq
        self.output = gamma * self.x_hat + beta
        self.avg = self.momentum * self.avg + (1 - self.momentum) * mu
        self.var = self.momentum * self.var + (1 - self.momentum) * var

    def partials(self, d_out):
        x_hat = self.x_hat
        dx_hat = self.gamma * d_out
        axis = self.sample_axis

        m = np.shape(d_out)[axis]
        p1 = m * dx_hat
        p2 = -np.sum(dx_hat, axis=axis, keepdims=True)
        p3 = - x_hat * np.sum(x_hat * dx_hat, axis=axis, keepdims=True)
        d_x = (p1 + p2 + p3) / (m * self.sq)

        d_beta = np.sum(d_out, axis=axis, keepdims=True)
        d_gamma = np.sum(x_hat * d_out, axis=axis, keepdims=True)

        return d_beta, d_gamma, d_x


class BatchNormPredict(Op):

    def __init__(self, train_op: BatchNormTraining = None,
                 avg: Tensor = None, var: Tensor = None, epsilon=1e-3):
        self.predefined_epsilon = None
        self.predefined_avg = None
        self.predefined_var = None
        self.train_op = None
        if avg is not None and var is not None:
            validate.is_tensor("batch_norm_avg", avg)
            validate.is_tensor("batch_norm_var", var)
            validate.is_not_none("batch_norm_epsilon", epsilon)
            self.predefined_epsilon = epsilon
            self.predefined_avg = avg
            self.predefined_var = var
        elif train_op is not None:
            validate.is_a("batch_norm_training", train_op, BatchNormTraining)
            self.train_op = train_op
        else:
            raise ValueError(
                "Incompatible combination of parameters, should either submit "
                "a BatchNormTraining op or submit avg and var constants")

    @classmethod
    def from_training(cls, train_op: BatchNormTraining):
        return cls(train_op)

    @classmethod
    def from_fixed_values(cls, avg: Tensor = None, var: Tensor = None):
        return cls(avg, var)

    @property
    def avg(self):
        if self.train_op is not None:
            return self.train_op.avg
        else:
            return self.predefined_avg

    @property
    def var(self):
        if self.train_op is not None:
            return self.train_op.var
        else:
            return self.predefined_var

    @avg.setter
    def avg(self, value):
        if self.train_op is not None:
            self.train_op.avg = value
        else:
            self.predefined_avg = value

    @var.setter
    def var(self, value):
        if self.train_op is not None:
            self.train_op.var = value
        else:
            self.predefined_var = value

    @property
    def epsilon(self):
        if self.train_op is not None:
            return self.train_op.epsilon
        else:
            return self.predefined_epsilon

    def check_incoming_shapes(self, *static_shapes):
        pass

    def compute_out_shape(self, beta_shape, gamma_shape,
                          x_shape: StaticShape) -> StaticShape:
        return x_shape

    def compute(self, beta: Tensor, gamma: Tensor, x: Tensor):
        sq = np.sqrt(self.var + self.epsilon)
        x_hat = (x - self.avg) / sq
        self.output = gamma * x_hat + beta
