import numpy as np

from chains.core.ops_activation import SoftMax
from chains.core.ops_losses import SoftMaxCrossEntropy
from chains.core.ops_mat import ArgMax
from .graph import Node
from .initializers import ConstantInitializer, VarInitializer
from .ops import Var, Placeholder, Constant
from .ops_activation import Sigmoid, TanH, ReLu
from .ops_arithmetic import IsGreaterThan
from .ops_losses import SigmoidCrossEntropy
from .ops_mat import MatMul, MaxComponent, SumComponents, AvgComponents
from .ops_mat import Transpose, Reshape, AsScalar, DimOp
from .ops_nn import FullyConnected
from .ops_regularization import L2NormRegularization, Dropout
from .shape import StaticShape

__all__ = ["initialized_var", "var", "placeholder", "constant", "add", "sub",
           "mul", "pow", "neg", "mat_mul", "mat_max", "mat_sum", "mat_avg",
           "transpose", "reshape", "as_scalar", "sigmoid_cross_entropy",
           "sigmoid", "is_greater_than", "tanh", "relu", "dim",
           "l2_norm_regularizer", "dropout", "fully_connected"]


def initialized_var(name: str, value):
    if name is None:
        raise ValueError('A variable must have a name')
    return Node(Var(initializer=ConstantInitializer(value),
                    shape=StaticShape.from_tuple(np.shape(value)),
                    dtype=np.array(value).dtype), name=name)


def var(name: str, initializer: VarInitializer, shape, dtype=np.float32):
    if name is None:
        raise ValueError('A variable must have a name')
    if not isinstance(initializer, VarInitializer):
        raise ValueError('Var should be passed a VarInitializer subclass')

    return Node(Var(initializer=initializer,
                    shape=StaticShape.from_tuple(shape),
                    dtype=dtype), name=name)


def placeholder(shape, dtype=np.float64):
    return Node(Placeholder(StaticShape.from_tuple(shape), dtype))


def constant(value):
    return Node(Constant(value))


def add(left: Node, right: Node):
    return left + right


def sub(left: Node, right: Node):
    return left - right


def mul(left: Node, right: Node):
    return left * right


def pow(value: Node, exp: int):
    return value ** exp


def neg(value: Node):
    return -value


def mat_mul(left: Node, right: Node):
    return Node(MatMul(), [left, right])


def mat_max(left: Node):
    return Node(MaxComponent(), [left])


def mat_sum(left: Node):
    return Node(SumComponents(), [left])


def mat_avg(left: Node):
    return Node(AvgComponents(), [left])


def transpose(left: Node):
    return Node(Transpose(), [left])


def reshape(shape: StaticShape, left: Node):
    return Node(Reshape(shape), [left])


def as_scalar(left: Node):
    return Node(AsScalar(), [left])


def softmax_cross_entropy(logits: Node, labels: Node):
    return Node(SoftMaxCrossEntropy(), [logits, labels])


def sigmoid_cross_entropy(logits: Node, labels: Node):
    return Node(SigmoidCrossEntropy(), [logits, labels])


def sigmoid(logits: Node):
    return Node(Sigmoid(), [logits])


def softmax(logits: Node):
    return Node(SoftMax(), [logits])


def is_greater_than(logits: Node, threshold: float):
    return Node(IsGreaterThan(threshold), [logits])


def argmax(logits: Node, axis=0):
    return Node(ArgMax(axis), [logits])


def tanh(logits: Node):
    return Node(TanH(), [logits])


def relu(logits: Node):
    return Node(ReLu(), [logits])


def dim(logits: Node, axis: int = -1):
    return Node(DimOp(axis), [logits])


def l2_norm_regularizer(lambd, batch_size, weight_matrices_array):
    if type(batch_size) == int:
        batch_size_node = Node(Constant(batch_size))
    elif type(batch_size) == Node:
        batch_size_node = batch_size
    return Node(L2NormRegularization(lambd=lambd),
                incoming_nodes=[batch_size_node] + weight_matrices_array)


def dropout(keep_prob, logits: Node):
    return Node(Dropout(keep_prob), [logits])


def fully_connected(inputs: Node, weights: Node, bias: Node,
                    *, first_layer=False) -> Node:
    return Node(FullyConnected(not first_layer), [inputs, weights, bias])
