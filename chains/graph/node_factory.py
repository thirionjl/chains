import numpy as np

import chains.graph.structure as g
import chains.initialization.variable_initializers as init
import chains.operations.activation_ops as activations
import chains.operations.arithmetic_ops as arithmetic
import chains.operations.base_ops as base
import chains.operations.losses_ops as losses
import chains.operations.mat_ops as mat
from chains.tensor.tensor import Shape


def initialized_var(name: str, value):
    if name is None:
        raise ValueError('A variable must have a name')
    return g.Node(base.Var(initializer=init.ConstantInitializer(value),
                           shape=Shape.from_tuple(np.shape(value)),
                           dtype=np.array(value).dtype),
                  name=name)


def var(name: str, initializer: init.VarInitializer, shape, dtype=np.float64):
    if name is None:
        raise ValueError('A variable must have a name')
    if not isinstance(initializer, init.VarInitializer):
        raise ValueError('Var should be passed a VarInitializer subclass')

    return g.Node(base.Var(initializer=initializer,
                           shape=Shape.from_tuple(shape),
                           dtype=dtype),
                  name=name)


def placeholder(shape, dtype=np.float64):
    return g.Node(base.Placeholder(Shape.from_tuple(shape), dtype))


def constant(value):
    return g.Node(base.Constant(value))


def add(left: g.Node, right: g.Node):
    return left + right


def sub(left: g.Node, right: g.Node):
    return left - right


def mul(left: g.Node, right: g.Node):
    return left * right


def pow(value: g.Node, exp: int):
    return value ** exp


def neg(value: g.Node):
    return -value


def mat_mul(left: g.Node, right: g.Node):
    return g.Node(mat.MatMul(), [left, right])


def mat_max(left: g.Node):
    return g.Node(mat.MaxComponent(), [left])


def mat_sum(left: g.Node):
    return g.Node(mat.SumComponents(), [left])


def mat_avg(left: g.Node):
    return g.Node(mat.AvgComponents(), [left])


def transpose(left: g.Node):
    return g.Node(mat.Transpose(), [left])


def reshape(shape: Shape, left: g.Node):
    return g.Node(mat.Reshape(shape), [left])


def as_scalar(left: g.Node):
    return g.Node(mat.AsScalar(), [left])


# def softmax_cross_entropy(logits: g.Node, labels: g.Node):
#     return g.Node(losses.SoftMaxCrossEntropyWithLogits(), [logits, labels])


def sigmoid_cross_entropy(logits: g.Node, labels: g.Node):
    return g.Node(losses.SigmoidCrossEntropy(), [logits, labels])


def sigmoid(logits: g.Node):
    return g.Node(activations.Sigmoid(), [logits])


def is_greater_than(logits: g.Node, threshold: float):
    return g.Node(arithmetic.IsGreaterThan(threshold), [logits])


def tanh(logits: g.Node):
    return g.Node(activations.TanH(), [logits])


def relu(logits: g.Node):
    return g.Node(activations.ReLu(), [logits])


def dim(logits: g.Node, axis: int = -1):
    return g.Node(mat.DimOp(axis), [logits])
