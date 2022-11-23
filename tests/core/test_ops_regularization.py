import numpy as np

from chains.core import ops_regularization as reg, env
from chains.core.static_shape import StaticShape


def test_coursera_dropout_forward():
    np.random.seed(1)
    dropout = reg.Dropout(keep_prob=0.7)
    dropout.compute(
        np.array(
            [
                [0.0, 3.32524635, 2.13994541, 2.60700654, 0.0],
                [0.0, 4.1600994, 0.79051021, 1.46493512, 0.0],
            ]
        )
    )

    np.testing.assert_equal(
        dropout.mask,
        np.array([[True, False, True, True, True], [True, True, True, True, True]]),
    )
    np.testing.assert_allclose(
        dropout.output,
        np.array(
            [
                [0.0, 0.0, 3.05706487, 3.72429505, 0.0],
                [0.0, 5.94299915, 1.1293003, 2.09276446, 0.0],
            ]
        ),
    )


def test_coursera_dropout_backward():
    np.random.seed(1)
    dropout = reg.Dropout(keep_prob=0.8)

    dropout.mask = np.array(
        [
            [True, False, True, False, True],
            [False, True, False, True, True],
            [False, False, True, False, False],
        ]
    )

    d = dropout.partials(
        np.array(
            [
                [0.46544685, 0.34576201, -0.00239743, 0.34576201, -0.22172585],
                [0.57248826, 0.42527883, -0.00294878, 0.42527883, -0.27271738],
                [0.45465921, 0.3377483, -0.00234186, 0.3377483, -0.21658692],
            ]
        )
    )[0]

    np.testing.assert_allclose(
        d,
        np.array(
            [
                [0.58180856, 0.0, -0.00299679, 0.0, -0.27715731],
                [0.0, 0.53159854, -0.0, 0.53159854, -0.34089673],
                [0.0, 0.0, -0.00292733, 0.0, -0.0],
            ]
        ),
        atol=1e-8,
    )


def test_l2_regularization_coursera_test_case():
    env.seed(1)
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
    # TODO Auto adjustment lamda = wanted_decay_rate_percent * m / learning_rate
    # wanted_decay_rate_percent = 0.1 (10%)

    w1 = np.array([[1, 2, 3], [1, 2, 3]])
    w2 = np.array([[1, 2], [3, 4]])
    lamda = 10.0
    batch_size = 32
    r = lamda / batch_size

    norm = reg.L2NormRegularization(lamda)
    norm.check_incoming_shapes(StaticShape.scalar(), StaticShape.from_tuple((1, 2)))
    norm.compute(batch_size, w1, w2)
    grad = norm.partials(1)

    np.testing.assert_equal(norm.output, 9.0625)
    np.testing.assert_allclose(grad[0], -norm.output / batch_size)
    np.testing.assert_allclose(grad[1], r * w1)
    np.testing.assert_allclose(grad[2], r * w2)


if __name__ == "__main__":
    test_l2_regularization_coursera_test_case()
