import numpy as np
import pytest

import chains.layers.conv.filter_utils as fu


def test_all_dimensions_without_error():
    X = np.zeros((32, 3, 500, 400))
    F = np.zeros((5, 3, 12, 10))
    assert fu.all_dimensions(X, F) == (32, 3, 500, 400, 5, 12, 10)


def test_all_dimensions_with_non_matching_channels():
    with pytest.raises(ValueError) as ex:
        fu.all_dimensions(np.zeros((32, 4, 500, 400)), np.zeros((5, 3, 12, 10)))
    assert str(ex.value) == 'Number of channels should be the same in activations(4) and filters(3)'


def test_im2col_1channel_1example_stride1():
    W = np.zeros((1, 1, 2, 2))
    X = np.arange(1, 10).reshape(1, 1, 3, 3)
    np.testing.assert_array_equal(fu.im2col_activations(X, W),
                                  np.array([[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]]))


def test_im2col_1channel_1example_stride2():
    W = np.zeros((1, 1, 2, 2))
    X = np.arange(1, 17).reshape(1, 1, 4, 4)
    np.testing.assert_array_equal(fu.im2col_activations(X, W, stride=2),
                                  np.array([[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]).T)


def test_im2col_multi_channel_multi_example_stride1():
    X1 = np.arange(1, 28).reshape(3, 3, 3)
    X = np.stack([X1, -X1])
    W = np.zeros((1, 3, 2, 2))
    E = np.array([[1, 2, 4, 5, 10, 11, 13, 14, 19, 20, 22, 23],
                  [2, 3, 5, 6, 11, 12, 14, 15, 20, 21, 23, 24],
                  [4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26],
                  [5, 6, 8, 9, 14, 15, 17, 18, 23, 24, 26, 27]])
    R = np.concatenate([E, -E]).T
    np.testing.assert_array_equal(fu.im2col_activations(X, W), R)


def test_im2col_back():
    X = np.zeros((2, 3, 3, 3), dtype=np.int64)
    F = np.zeros((1, 3, 2, 2))
    XC1 = np.arange(1, 49).reshape(4, 12).T
    XC = np.hstack([XC1, -XC1])
    X1 = np.array([
        [[1, 15, 14], [28, 82, 54], [27, 67, 40]],
        [[5, 23, 18], [36, 98, 62], [31, 75, 44]],
        [[9, 31, 22], [44, 114, 70], [35, 83, 48]]
    ])
    X_back_expected = np.stack([X1, -X1])
    np.testing.assert_array_equal(X_back_expected, fu.im2col_activations_back(XC, X, F))


def test_reshape_out():
    ZC = np.arange(1, 17).reshape(2, 2 * 2 * 2)
    X = np.random.randint(1, 10, (2, 3, 3, 3), dtype=ZC.dtype)
    F = np.random.randint(1, 10, (2, 3, 2, 2), dtype=ZC.dtype)
    Z = fu.reshape_out(ZC, X, F)
    np.testing.assert_array_equal(Z, np.array([
        [[[1, 2], [3, 4]], [[9, 10], [11, 12]]],
        [[[5, 6], [7, 8]], [[13, 14], [15, 16]]]
    ]))


def test_reshape_out_back_and_forth():
    ZC = np.arange(1, 17).reshape(2, 2 * 2 * 2)
    X = np.random.randint(1, 10, (2, 3, 3, 3), dtype=ZC.dtype)
    F = np.random.randint(1, 10, (2, 3, 2, 2), dtype=ZC.dtype)
    Z = fu.reshape_out(ZC, X, F)
    ZC_back = fu.reshape_out_back(Z, X, F)
    np.testing.assert_array_equal(ZC, ZC_back)


def test_im2col_filter_back_and_forth():
    X = np.random.randint(1, 10, (2, 3, 3, 3))
    F = np.random.randint(1, 10, (2, 3, 2, 2))
    FC = fu.im2col_filters(X, F)
    F_back = fu.im2col_filters_back(FC, X, F)
    np.testing.assert_array_equal(F, F_back)
