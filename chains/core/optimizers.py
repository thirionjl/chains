import abc
from typing import Dict

import numpy as np

from .graph import Graph, Node
from .shape import StaticShape
from .tensor import Tensor

__all__ = ["Optimizer", "GradientDescentOptimizer", "MomentumOptimizer",
           "AdamOptimizer"]


# TODO Get back learning rate finder...
class Optimizer(abc.ABC):

    def __init__(self, learning_rate: float):
        self.lr = learning_rate
        self._graph = None
        self.cost = None

    def initialize_and_check(self, graph: Graph):
        if not isinstance(graph, Graph):
            raise ValueError(
                f"Optimizer should be initialized with a {Graph}"
                f", but got a {type(graph)}")
        self._graph = graph
        self._graph.check_initialized()
        self._graph.forward()
        self.cost = None

    def run(self):
        c = self._graph.forward()
        gradient = self._graph.backward()
        self.apply_gradient(gradient)
        self.cost = c
        return gradient, c

    @abc.abstractmethod
    def apply_gradient(self, gradient: Dict[Node, Tensor]):
        pass


class GradientDescentOptimizer(Optimizer):

    def apply_gradient(self, grads: Dict[Node, Tensor]):
        for var_node in self._graph.variables:
            var_node.value -= self.lr * grads[var_node]


class MomentumOptimizer(Optimizer):

    def __init__(self, lr, beta=0.9):
        super().__init__(lr)
        _check_is_percentage(beta, "Beta")
        self.beta = beta
        self.v = {}  # velocities

    def initialize_and_check(self, graph: Graph):
        super().initialize_and_check(graph)
        _initialize_zeros(graph, self.v)

    def apply_gradient(self, grads: Dict[Node, Tensor]):
        for n in self._graph.variables:
            self.v[n] = self.beta * self.v[n] + (1 - self.beta) * grads[n]
            n.value -= self.lr * self.v[n]


class AdamOptimizer(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        _check_is_percentage(beta1, "Beta1")
        _check_is_percentage(beta2, "Beta2")
        if not 0 < epsilon < 1e-6:
            raise ValueError(f"Epsilon should be less that 1e-6 got {epsilon}")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.beta1_pow = 1
        self.beta2_pow = 1
        self.v = {}  # velocities
        self.s = {}  # squared velocities
        self.t = 0

    def initialize_and_check(self, graph: Graph):
        super().initialize_and_check(graph)
        _initialize_zeros(graph, self.v)
        _initialize_zeros(graph, self.s)

    def apply_gradient(self, grads: Dict[Node, Tensor]):
        self.t += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        for n in self._graph.variables:
            g_squared = grads[n] ** 2
            self.v[n] = self.beta1 * self.v[n] + (1 - self.beta1) * grads[n]
            self.s[n] = self.beta2 * self.s[n] + (1 - self.beta2) * g_squared
            v_cor = self.v[n] / (1 - self.beta1_pow)
            s_cor = self.s[n] / (1 - self.beta2_pow)

            n.value -= self.lr * v_cor / np.sqrt(s_cor + self.epsilon)


def _check_has_known_shape(n: StaticShape):
    static_shape = n.shape
    if static_shape.is_unknown():
        raise ValueError(f"Variable node {n} should have a static "
                         f"shape without unknown dimensions, but got "
                         f"has dimensions {static_shape}")


def _check_is_percentage(f: float, parameter_name: str):
    if not 0 <= f < 1:
        raise ValueError(f"{parameter_name} should be >=0 and <1, but got {f}")


def _initialize_zeros(g: Graph, velocity_map: Dict[Node, Tensor]):
    for var_node in g.variables:
        _check_has_known_shape(var_node)
        velocity_map[var_node] = np.zeros(var_node.shape.to_numpy())
