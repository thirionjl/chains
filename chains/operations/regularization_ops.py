import numpy as np

import chains.graph.node_factory as nf
import chains.graph.structure as g
from chains.operations.base_ops import Op, ElementWiseUnaryOp
from chains.tensor.tensor import Tensor, Shape


def l2_norm_regularizer(lambd, batch_size, weight_matrices_array):
    if type(batch_size) == int:
        batch_size_node = nf.constant(batch_size)
    elif type(batch_size) == g.Node:
        batch_size_node = batch_size
    return g.Node(L2NormRegularization(lambd=lambd),
                  incoming_nodes=[batch_size_node] + weight_matrices_array)


def dropout(keep_prob, logits: g.Node):
    return g.Node(Dropout(keep_prob), [logits])


class L2NormRegularization(Op):

    def __init__(self, lambd: float, epsilon=1e-12):
        super().__init__()
        if lambd is None:
            raise ValueError("L2NormRegularization parameter is mandatory")
        if type(lambd) != float:
            raise ValueError(
                f"L2NormRegularization parameter should be a float, "
                f"got {type(lambd)}")

        self.lamda = lambd
        self.epsilon = epsilon
        self.weights = None
        self.factor = None
        self.batch_size = None

    def check_incoming_shapes(self, batch_size: Shape, *weight_shapes):
        if not batch_size.is_scalar:
            raise ValueError("First argument must be scalar")

    def compute_out_shape(self, *args) -> Shape:
        return Shape.scalar()

    def compute(self, batch_size, *weights: tuple):
        self.batch_size = batch_size
        self.weights = weights
        self.factor = self.lamda / self.batch_size
        s = 0
        for x in weights:
            xr = np.ravel(x)
            s += np.dot(xr, xr)

        self.output = max(s, self.epsilon) * (self.factor / 2)

    def partials(self, d_output):
        return [-self.output * d_output / self.batch_size] + \
               [self.factor * w * d_output for w in self.weights]


class Dropout(ElementWiseUnaryOp):

    def __init__(self, keep_prob):
        if type(keep_prob) != float or not (0 < keep_prob <= 1):
            raise ValueError(f"Drop-out keep probability should be a float "
                             f"between 0 and 1, got {keep_prob}")
        self.keep_prob = keep_prob
        self.mask = None

    def compute_out_shape(self, x_shape: Shape) -> Shape:
        return x_shape

    def compute(self, x: Tensor):
        super().compute(x)
        self.mask = np.random.random_sample(np.shape(x)) < self.keep_prob
        self.output = self.mask * self.x / self.keep_prob

    def simple_derivative(self):
        return self.mask / self.keep_prob
