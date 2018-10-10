import numpy as np

import chains.graph.structure as g
import chains.operations.base_ops as ops
from chains.tensor.tensor import Tensor, Shape


def fully_connected(inputs: g.Node, weights: g.Node, bias: g.Node,
                    *, first_layer=False) -> g.Node:
    return g.Node(FullyConnected(not first_layer), [inputs, weights, bias])


class FullyConnected(ops.Op):

    def __init__(self, feature_derivative=True):
        super().__init__()
        self.bias_shape = None
        self.features = None
        self.weights = None
        self.bias = None
        self.feature_derivative = feature_derivative

    @staticmethod
    def check_incoming_shapes(f: Shape, w: Shape, b: Shape):
        if f.ndim != 2:
            raise ValueError("Inputs of a fully connected layer should be "
                             "a 2-D matrix")
        if w.ndim != 2:
            raise ValueError("Inputs of a fully connected layer should be "
                             "a 2-D matrix")
        if not (b.is_column()):
            raise ValueError(f"Bias should be a 2-D column vector, got {b}")

        if w[1] != f[0]:
            raise ValueError(f"Number of columns of weight matrix should match"
                             f" number of rows of inputs matrix \
            but got {w[1]} and {f[0]}")
        if w[0] != b[0]:
            raise ValueError(f"Number of rows of weight matrix should match "
                             f"first dimension of bias \
            but got {w[0]} and {b[0]}")

    @staticmethod
    def compute_out_shape(f: Shape, w: Shape, b: Shape) -> Shape:
        return Shape(w[0], f[1])

    def compute(self, features: Tensor, weights: Tensor, bias: Tensor):
        self.bias_shape = np.shape(bias)
        self.features = features
        self.weights = weights
        self.bias = bias

        self.output = weights @ features + bias  # Fixme: Broadcasting...

    def partials(self, d_output):
        d_bias = np.sum(d_output, axis=1, keepdims=True)
        d_weights = d_output @ self.features.T
        d_features = self.weights.T @ d_output if self.feature_derivative \
            else 0
        # np.reshape(d_bias, self.bias_shape)
        return d_features, d_weights, d_bias
