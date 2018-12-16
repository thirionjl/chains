import numpy as np

from .graph import Node
from .initializers import ConstantInitializer, VarInitializer
from .ops import Var, Placeholder, Constant
from .ops_activation import SoftMax, LeakyReLu, Sigmoid, TanH, ReLu
from .ops_arithmetic import IsGreaterThan
from .ops_fully_connected import FullyConnected
from .ops_losses import SigmoidCrossEntropy, SoftMaxCrossEntropy
from .ops_mat import ArgMax, MatMul, MaxComponent, SumComponents, AvgComponents
from .ops_mat import Transpose, Reshape, AsScalar, DimOp
from .ops_norm import BatchNormTraining, BatchNormPredict
from .ops_regularization import L2NormRegularization, Dropout
from .static_shape import StaticShape
from .tensor import Tensor
from ..utils import validate

__all__ = ["initialized_var", "var", "placeholder", "constant", "add", "sub",
           "mul", "pow", "neg", "mat_mul", "mat_max", "mat_sum", "mat_avg",
           "transpose", "reshape", "as_scalar", "sigmoid_cross_entropy",
           "sigmoid", "is_greater_than", "tanh", "relu", "dim",
           "l2_norm_regularizer", "dropout", "fully_connected"]


def initialized_var(name: str, value):
    validate.is_not_blank("var_name", name)
    validate.is_tensor("var_value", value)
    return Node(Var(initializer=ConstantInitializer(value),
                    shape=np.shape(value),
                    dtype=np.array(value).dtype), name=name)


def var(name: str, initializer: VarInitializer, shape, dtype=np.float32):
    validate.is_not_blank("var_name", name)
    validate.is_a("var_shape", shape, tuple)

    return Node(Var(initializer=initializer,
                    shape=shape,
                    dtype=dtype), name=name)


def placeholder(shape, dtype=np.float32, name=None):
    return Node(Placeholder(shape, dtype), name=name)


def constant(value, dtype=np.float32, name=None):
    return Node(Constant(value, dtype), name=name)


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


def mat_mul(left: Node, right: Node, name=None):
    return Node(MatMul(), [left, right], name=name)


def mat_max(left: Node, name=None):
    return Node(MaxComponent(), [left], name=name)


def mat_sum(left: Node, name=None):
    return Node(SumComponents(), [left], name=name)


def mat_avg(left: Node, name=None):
    return Node(AvgComponents(), [left], name=name)


def transpose(left: Node, name=None):
    return Node(Transpose(), [left], name=name)


def reshape(shape: StaticShape, left: Node, name=None):
    return Node(Reshape(shape), [left], name=name)


def as_scalar(left: Node, name=None):
    return Node(AsScalar(), [left], name=name)


def softmax_cross_entropy(logits: Node, labels: Node, name=None):
    return Node(SoftMaxCrossEntropy(), [logits, labels], name=name)


def sigmoid_cross_entropy(logits: Node, labels: Node, name=None):
    return Node(SigmoidCrossEntropy(), [logits, labels], name=name)


def sigmoid(logits: Node, name=None):
    return Node(Sigmoid(), [logits], name=name)


def softmax(logits: Node, name=None):
    return Node(SoftMax(), [logits], name=name)


def is_greater_than(logits: Node, threshold: float, name=None):
    return Node(IsGreaterThan(threshold), [logits], name=name)


def argmax(logits: Node, axis=0, name=None):
    return Node(ArgMax(axis), [logits], name=name)


def tanh(logits: Node, name=None):
    return Node(TanH(), [logits], name=name)


def relu(logits: Node, name=None):
    return Node(ReLu(), [logits], name=name)


def leaky_relu(logits: Node, name=None):
    return Node(LeakyReLu(), [logits], name=name)


def dim(logits: Node, axis: int = -1, name=None):
    return Node(DimOp(axis), [logits], name=name)


def l2_norm_regularizer(lambd, batch_size, weight_matrices_array, name=None):
    if isinstance(batch_size, int):
        batch_size_node = Node(Constant(batch_size))
    elif isinstance(batch_size, Node):
        batch_size_node = batch_size
    return Node(L2NormRegularization(lambd=lambd),
                incoming_nodes=[batch_size_node] + weight_matrices_array,
                name=name)


def dropout(keep_prob, logits: Node, name=None):
    return Node(Dropout(keep_prob), [logits], name=name)


def fully_connected(inputs: Node, weights: Node, bias: Node,
                    *, first_layer=False, name=None) -> Node:
    return Node(FullyConnected(not first_layer), [inputs, weights, bias],
                name=name)


def batch_norm_train(inputs: Node, beta: Node, gamma: Node, momentum=0.99,
                     epsilon=1e-3, sample_axis=-1, name=None) -> Node:
    bn = BatchNormTraining(momentum, epsilon, sample_axis)
    return Node(bn, [beta, gamma, inputs], name=name)


def batch_norm_predict(batch_norm: Node, inputs: Node, beta: Node,
                       gamma: Node, name=None) -> Node:
    bn = BatchNormPredict.from_training(batch_norm)
    return Node(bn, [beta, gamma, inputs], name=name)


def batch_norm_predict_fixed(mean: Tensor, variance: Tensor, inputs: Node,
                             beta: Node,
                             gamma: Node, name=None) -> Node:
    bn = BatchNormPredict.from_fixed_values(mean, variance)
    return Node(bn, [beta, gamma, inputs], name=name)
