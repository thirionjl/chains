import numpy as np

from chains.core.ops_activation import SoftMax, LeakyReLu
from chains.core.ops_losses import SoftMaxCrossEntropy
from chains.core.ops_mat import ArgMax
from chains.core.ops_norm import BatchNormTraining, BatchNormPredict
from chains.core.tensor import Tensor
from chains.utils import validate
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


def placeholder(shape, dtype=np.float32):
    return Node(Placeholder(shape, dtype))


def constant(value, dtype=np.float32):
    return Node(Constant(value, dtype))


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


def leaky_relu(logits: Node):
    return Node(LeakyReLu(), [logits])


def dim(logits: Node, axis: int = -1):
    return Node(DimOp(axis), [logits])


def l2_norm_regularizer(lambd, batch_size, weight_matrices_array):
    if isinstance(batch_size, int):
        batch_size_node = Node(Constant(batch_size))
    elif isinstance(batch_size, Node):
        batch_size_node = batch_size
    return Node(L2NormRegularization(lambd=lambd),
                incoming_nodes=[batch_size_node] + weight_matrices_array)


def dropout(keep_prob, logits: Node):
    return Node(Dropout(keep_prob), [logits])


def fully_connected(inputs: Node, weights: Node, bias: Node,
                    *, first_layer=False) -> Node:
    return Node(FullyConnected(not first_layer), [inputs, weights, bias])


def batch_norm_train(inputs: Node, beta: Node, gamma: Node, momentum=0.99,
                     epsilon=1e-3, sample_axis=-1) -> Node:
    bn = BatchNormTraining(momentum, epsilon, sample_axis)
    return Node(bn, [beta, gamma, inputs])


def batch_norm_predict(batch_norm: Node, inputs: Node, beta: Node,
                       gamma: Node) -> Node:
    bn = BatchNormPredict.from_training(batch_norm)
    return Node(bn, [beta, gamma, inputs])


def batch_norm_predict_fixed(mean: Tensor, variance: Tensor, inputs: Node,
                             beta: Node,
                             gamma: Node) -> Node:
    bn = BatchNormPredict.from_fixed_values(mean, variance)
    return Node(bn, [beta, gamma, inputs])
