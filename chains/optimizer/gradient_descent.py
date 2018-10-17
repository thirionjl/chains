import abc
from typing import Dict

import chains.graph.structure as g
from chains.tensor.tensor import Tensor


# TODO Get back learning rate finder...
class Optimizer(abc.ABC):

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self._graph = None
        self.cost = None

    def initialize_and_check(self, graph: g.Graph):
        if not isinstance(graph, g.Graph):
            raise ValueError(f"Optimizer should be initialized with a {g.Graph}"
                             f", but got a {type(graph)}")
        self._graph = graph
        self._graph.check_initialized()
        self._graph.forward()
        self.cost = None

    def run(self):
        c = self._graph.forward()
        gradient = self._graph.backward()
        self.apply_gradient(gradient, self.learning_rate)
        self.cost = c
        return gradient, c

    @abc.abstractmethod
    def apply_gradient(self, gradient: Dict[g.Node, Tensor],
                       learning_rate: float):
        pass


class GradientDescentOptimizer(Optimizer):

    def apply_gradient(self, grads: Dict[g.Node, Tensor], lr: float):
        for var_node in self._graph.variables:
            var_node.value += - lr * grads[var_node]

# TODO Momentum
# TODO Adam, RMSProp
