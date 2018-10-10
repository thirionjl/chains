import numpy as np

from chains.preprocessing.inputs import normalize


def test_normalize():
    A = np.arange(start=1, stop=7).reshape(2, 3)

    np.testing.assert_allclose(normalize(A, 0), np.array([[-1., -1., -1.], [1., 1., 1.]]))
    np.testing.assert_allclose(normalize(A, 1),
                               np.array([[-1.22474487, 0., 1.22474487], [-1.22474487, 0., 1.22474487]]))
