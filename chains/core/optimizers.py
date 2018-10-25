import abc
import math
from typing import Dict

import numpy as np

from chains.utils import validate
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
        self._variables = None
        self.multiplier = 1

    def initialize_and_check(self, graph: Graph):
        validate.is_a("graph", graph, Graph)
        if not graph.shape.is_scalar():
            raise ValueError(f"Optimizers accept only graphs that output a "
                             f"scalar, but got {self.shape}")
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

    def apply_gradients(self, grads: Dict[Node, Tensor]):
        for var_node in self._variables:
            var_node.value = self.apply_single_gradient(var_node,
                                                        grads[var_node])

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

    def initialize_and_check(self, graph: Graph):
        super().initialize_and_check(graph)
        _initialize_zeros(graph, self.v)

    def apply_single_gradient(self, var, grad):
        velocity = self.v[var]
        velocity = self.beta * velocity + (1 - self.beta) * grad
        self.v[var] = velocity
        return var.value - self.lr * grad


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

    def apply_gradients(self, grads: Dict[Node, Tensor]):
        self.t += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2
        for n in self._graph.variables:
            g = grads[n]
            v = self.v[n]
            s = self.s[n]
            g2 = g * g

            v = self.vmean(g, v, n)
            s = self.smean(g2, s)
            lr_adj = self.adjust_lr()
            velocity = self.divide_velocity(s, v)
            n.value = self.apply_velocity(lr_adj, n, velocity)
            self.v[n] = v
            self.s[n] = s

            if v.dtype != np.float32 or s.dtype != np.float32 or n.value.dtype != np.float32:
                print("Changed dtype automatically ?")
                variables = {'g': g, 'g2': g2, 'v': v, 's': s,
                             'velocity': velocity, 'n_value': n.value}
                for key, value in variables.items():
                    print(f"{key}.dtype = {value.dtype}")
                raise RuntimeError("Auto switch to float64")

    def apply_velocity(self, lr_adj, n, velocity):
        return n.value - lr_adj * velocity

    def divide_velocity(self, s, v):
        return v / (np.sqrt(s) + self.epsilon)

    def adjust_lr(self):
        return self.lr * math.sqrt(1 - self.beta2_pow) / (1 - self.beta1_pow)

    def smean(self, g2, s):
        return self.beta2 * s + (1 - self.beta2) * g2

    def vmean(self, g, v, n):

        beta_g = (1 - self.beta1) * g

        try:
            return self.beta1 * v + beta_g
        except FloatingPointError:
            print(f"Break for {n.name}")
            raise


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
        velocity_map[var_node] = 0
