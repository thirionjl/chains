import numpy as np

from chains.operations import regularization_ops as reg
from chains.tensor.tensor import Shape


def test_dropout():
    dropout = reg.Dropout(keep_prob=0.5, seed=1)
    dropout.compute(np.array([[1, 2, 3], [4, 5, 6]]))
    gradient = dropout.partials(1)[0]

    np.testing.assert_equal(dropout.output, np.array([[1, 2, 0], [0, 5, 6]]))
    np.testing.assert_equal(gradient, np.array([[1, 1, 0], [0, 1, 1]]))


def test_l2_regularization_coursera_test_case():
    np.random.seed(1)
    y_assess = np.array([[1, 1, 0, 1, 0]])
    w1 = np.random.randn(2, 3)
    np.random.randn(2, 1)
    w2 = np.random.randn(3, 2)
    np.random.randn(3, 1)
    w3 = np.random.randn(1, 3)
    np.random.randn(1, 1)

    norm = reg.L2NormRegularization(0.1)
    norm.compute(y_assess.shape[-1], w1, w2, w3)

    np.testing.assert_allclose(norm.output, 0.183984340402)


def test_l2_regularization():
    # TODO Auto adjustment lamda = wanted_decay_rate_percent * m / learning_rate # wanted_decay_rate_percent = 0.1 (10%)

    w1 = np.array([[1, 2, 3], [1, 2, 3]])
    w2 = np.array([[1, 2], [3, 4]])
    lamda = 10.0
    batch_size = 32
    r = lamda / batch_size

    norm = reg.L2NormRegularization(lamda)
    norm.check_incoming_shapes(Shape.scalar(), Shape.from_tuple((1, 2)))
    norm.compute(batch_size, w1, w2)
    grad = norm.partials(1)

    np.testing.assert_equal(norm.output, 9.0625)
    np.testing.assert_allclose(grad[0], - norm.output / batch_size)
    np.testing.assert_allclose(grad[1], r * w1)
    np.testing.assert_allclose(grad[2], r * w2)


if __name__ == '__main__':
    test_l2_regularization_coursera_test_case()
