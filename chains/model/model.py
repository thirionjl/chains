from abc import ABC
from collections import namedtuple
from typing import Iterable

import chains.layers.fully_connected as fc
from chains.graph import node_factory as f
from chains.graph.structure import Node
from chains.initialization import variable_initializers as init
from chains.operations import regularization_ops as reg
from chains.tensor.tensor import Dim


class Model(ABC):

    def __init__(self, optimizer, cost_graph, prediction_graph):
        self.cost_graph = None
        self.prediction_graph = None

    def train(self, x_train, y_train, *, num_iterations=2_500, learning_rate=0.0075, print_cost=False):
        pass

    def predict(self, x_test):
        self.prediction_graph.placeholders = {self.X: x_test}
        return self.prediction_graph.evaluate()


class GraphBuilder(ABC):

    def __init__(self, use_for_predictions=True, use_for_costs=True):
        self.use_for_predictions = use_for_predictions
        self.use_for_costs = use_for_costs

    def __call__(self, pos: int, input_graph: Node):
        raise NotImplementedError()


def sequence(prediction_graph: bool, cost_graph: bool, features, builders: Iterable[GraphBuilder]):
    filtered = filter(lambda b: b.use_for_predictions == prediction_graph and b.use_for_costs == cost_graph, builders)

    graph = features
    for pos, builder in filtered:
        graph = builder(pos, graph)

    return graph


class Features:
    def __init__(self, cnt_features: int):
        super().__init__()
        self.cnt_features = Dim.of(cnt_features)
        self.cnt_examples = Dim.unknown()

    def __call__(self, *args):
        return f.placeholder(shape=(self.cnt_features, self.cnt_examples))  # TODO Allow axis swap


class FullyConnectedLayer:
    Activation = namedtuple("Activation", ["initializer", "builder"])

    activations = {
        None: Activation(initializer=init.XavierInitializer, builder=lambda x: x),
        "tanh": Activation(initializer=init.XavierInitializer, builder=f.tanh),
        "relu": Activation(initializer=init.HeInitializer, builder=f.relu),
        "sigmoid": Activation(initializer=init.XavierInitializer, builder=f.sigmoid),
    }

    def __init__(self, neurons: int, activation: str, weight_initializer=None, bias_initializer=None):
        self._check_activation(activation)
        self.activation = activation
        self.cnt_neurons = neurons
        self.weight_initializer = self._weight_init(activation) if weight_initializer is None else weight_initializer
        self.bias_initializer = init.ZeroInitializer if bias_initializer is None else bias_initializer
        self.weights = None
        self.bias = None
        self.linear = None
        self.output = None

    @classmethod
    def _check_activation(cls, a):
        keys = cls.activations.keys()
        if a not in keys:
            raise ValueError(f"Invalid activation function argument, should be one of {keys}, but got {a}")

    @classmethod
    def _weight_init(cls, a):
        return cls.activations[a].initializer

    def __call__(self, pos, features: Node):
        cnt_features = features.shape[0]  # TODO Allow different axis for examples
        self.weights = f.var("W" + str(pos + 1), self.weight_initializer, shape=(self.cnt_neurons, cnt_features))
        self.biases = f.var("b" + str(pos + 1), self.bias_initializer, shape=(self.cnt_neurons, 1))
        self.linear = fc.fully_connected(features, self.weights, self.biases, first_layer=(pos == 0))
        builder = self.activations[self.activation].builder
        return builder(self.linear)


class Dropout:
    def __init__(self, keep_prob=0.8):
        if not (0 < keep_prob <= 1):
            raise ValueError(f"Keep probability should be between 0 and 1")
            self.keep_prob = keep_prob

    def __call__(self, pos, features: Node):
        return reg.dropout(self.keep_prob, features)
