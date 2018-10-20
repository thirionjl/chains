import numpy as np

import chains.todo.conv.layer as cl


def test_simple_forward_prop():
    X = np.arange(1, 10).reshape(1, 1, 3, 3)
    F = np.arange(4, 0, -1).reshape(1, 1, 2, 2)
    b = np.zeros((1, 1), dtype=np.int64)
    Z, _, _, _ = cl.forward(X, F, b)
    np.testing.assert_array_equal(Z, np.array([23, 33, 53, 63]).reshape(1, 1, 2, 2))


def test_forward_backward_dims():
    X = np.arange(1, 10).reshape(1, 1, 3, 3)
    F = np.arange(4, 0, -1).reshape(1, 1, 2, 2)
    b = np.zeros((1, 1), dtype=np.int64)
    Z, XC, FC, ZC = cl.forward(X, F, b)
    dZ = Z
    dX, dF, db = cl.backward(dZ, X, XC, F, FC, b)
    assert X.shape == dX.shape
    assert b.shape == db.shape
    assert F.shape == dF.shape
