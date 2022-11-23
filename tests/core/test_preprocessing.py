import numpy as np

from chains.core.preprocessing import normalize, one_hot


def test_normalize():
    A = np.arange(start=1, stop=7).reshape(2, 3)

    np.testing.assert_allclose(
        normalize(A, 0), np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    )
    np.testing.assert_allclose(
        normalize(A, 1),
        np.array([[-1.22474487, 0.0, 1.22474487], [-1.22474487, 0.0, 1.22474487]]),
    )


def test_one_hot():
    labels = np.array([1, 2, 3, 0, 2, 1])

    actual = one_hot(labels, cnt_classes=4)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(actual, expected)
