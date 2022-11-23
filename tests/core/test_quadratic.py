import numpy as np

import chains.core.node_factory as nf
import chains.core.optimizers as gd
from chains.core.graph import Graph


def test_quadratic():
    x = nf.initialized_var("x", np.array([[0.07], [0.4], [0.1]]))
    A = nf.constant(np.array([[2, -1, -2], [-1, 1, 0], [-2, 0, 4]]))
    B = nf.constant(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    b = nf.placeholder(shape=(3, 1))

    quadratic = x.T() @ A @ x
    linear = B @ x - b
    squared_linear = nf.mat_sum(linear) ** 2

    expr = quadratic + squared_linear

    cost_function = nf.as_scalar(expr) + 7

    cost = Graph(cost_function)
    cost.placeholders = {b: np.array([1, 0, 0]).reshape(3, 1).astype("float32")}
    cost.initialize_variables()
    optimizer = gd.GradientDescentOptimizer(0.1)
    optimizer.prepare_and_check(cost)

    for i in range(500):
        optimizer.run()

    np.testing.assert_allclose(cost.evaluate(), 7.0)
    np.testing.assert_allclose(x.value, np.array([[1.0], [1.0], [0.5]]))
