import abc
import math
from collections.abc import Mapping
from typing import Dict

import numpy as np

from chains.utils.nd_typing import NdArrayLike
from .graph import Graph, Node
from ..utils import validate

__all__ = [
    "Optimizer",
    "GradientDescentOptimizer",
    "MomentumOptimizer",
    "AdamOptimizer",
]


class Optimizer(abc.ABC):
    def __init__(self, learning_rate: float):
        self.lr = learning_rate
        self._graph = None
        self.cost = None
        self._variables = None
        self.multiplier = 1

    def prepare_and_check(self, graph: Graph):
        validate.is_a("graph", graph, Graph)
        if not graph.shape.is_scalar():
            raise ValueError(
                f"Optimizers accept only graphs that output a scalar, but got {graph.shape}"
            )
        self._graph = graph
        self._graph.check_initialized()
        self._graph.forward()
        self._variables = list(self._graph.variables)
        self.cost = None

    def run(self):
        c = self._graph.forward()
        gradient = self._graph.backward()
        self.apply_gradients(gradient)
        self.cost = c
        return gradient, c

    def apply_gradients(self, grads: Mapping[Node, NdArrayLike]):
        for var_node in self._variables:
            var_node.value = self.apply_single_gradient(var_node, grads[var_node])

    def apply_single_gradient(self, var, grad):
        pass


class GradientDescentOptimizer(Optimizer):
    def apply_single_gradient(self, var, grad):
        return var.value - self.lr * grad


class MomentumOptimizer(Optimizer):
    def __init__(self, lr, beta=0.9):
        super().__init__(lr)
        _check_is_percentage(beta, "Beta")
        self.beta = beta
        self.v = {}  # velocities

    def prepare_and_check(self, graph: Graph):
        super().prepare_and_check(graph)
        _initialize_zeros(graph, self.v)

    def apply_single_gradient(self, var, grad):
        velocity = self.beta * self.v[var] + (1 - self.beta) * grad
        self.v[var] = velocity
        return var.value - self.lr * velocity


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

    def prepare_and_check(self, graph: Graph):
        super().prepare_and_check(graph)
        _initialize_zeros(graph, self.v)
        _initialize_zeros(graph, self.s)

    def apply_gradients(self, grads: Dict[Node, NdArrayLike]):
        self.t += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        for n in self._graph.variables:
            g = grads[n] + 1e-10
            v = self.v[n]
            s = self.s[n]
            g2 = g * g

            v = self.beta1 * v + (1 - self.beta1) * g
            s = self.beta2 * s + (1 - self.beta2) * g2
            lr_adj = self.lr * math.sqrt(1 - self.beta2_pow) / (1 - self.beta1_pow)
            velocity = v / (np.sqrt(s) + self.epsilon)
            n.value = n.value - lr_adj * velocity
            self.v[n] = v
            self.s[n] = s


def _check_has_known_shape(n: Node):
    static_shape = n.shape
    if static_shape.has_unknown_dim():
        raise ValueError(
            f"Variable node {n} should have a static shape without unknown dimensions, but got {static_shape}"
        )


def _check_is_percentage(f: float, parameter_name: str):
    if not 0 <= f < 1:
        raise ValueError(f"{parameter_name} should be >=0 and <1, but got {f}")


def _initialize_zeros(g: Graph, velocity_map: Dict[Node, NdArrayLike]):
    for var_node in g.variables:
        _check_has_known_shape(var_node)
        velocity_map[var_node] = 0
