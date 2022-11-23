import numpy as np
import pytest

import chains.core.utils_conv as uc


def test_all_dimensions_without_error():
    inputs = (32, 3, 500, 400)
    filters = (5, 3, 12, 10)
    assert uc._all_dimensions(inputs, filters) == (32, 3, 500, 400, 5, 12, 10)


def test_all_dimensions_with_non_matching_channels():
    with pytest.raises(ValueError) as ex:
        uc._all_dimensions((32, 4, 500, 400), (5, 3, 12, 10))
    assert (
        str(ex.value) == "Number of channels should be the same "
        "in activations_shape(4) and filters_shape(3)"
    )


def test_im2col_1channel_1example_stride1():
    inputs = np.arange(1, 10).reshape(1, 1, 3, 3)
    filters_shape = (1, 1, 2, 2)
    np.testing.assert_array_equal(
        uc.im2col(inputs, filters_shape),
        np.array([[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]]),
    )


def test_im2col_same_convolution():
    inputs = np.arange(1, 17).reshape(1, 1, 4, 4)
    filters_shape = (1, 1, 3, 3)

    actual = uc.im2col(inputs, filters_shape, padding=1)

    np.testing.assert_array_equal(actual[:, 0], np.array([0, 0, 0, 0, 1, 2, 0, 5, 6]).T)
    np.testing.assert_array_equal(
        actual[:, 5], np.array([1, 2, 3, 5, 6, 7, 9, 10, 11]).T
    )
    np.testing.assert_array_equal(
        actual[:, -1], np.array([11, 12, 0, 15, 16, 0, 0, 0, 0]).T
    )


def test_im2col_1channel_1example_stride2():
    inputs = np.arange(1, 17).reshape(1, 1, 4, 4)
    filters_shape = (1, 1, 2, 2)
    np.testing.assert_array_equal(
        uc.im2col(inputs, filters_shape, stride=2),
        np.array([[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]).T,
    )


def test_im2col_multi_channel_multi_example_stride1():
    partial_input = np.arange(1, 28).reshape(3, 3, 3)
    inputs = np.stack([partial_input, -partial_input])
    assert inputs.shape == (2, 3, 3, 3)
    filters_shape = (1, 3, 2, 2)
    e = np.array(
        [
            [1, 2, 4, 5, 10, 11, 13, 14, 19, 20, 22, 23],
            [2, 3, 5, 6, 11, 12, 14, 15, 20, 21, 23, 24],
            [4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26],
            [5, 6, 8, 9, 14, 15, 17, 18, 23, 24, 26, 27],
        ]
    )
    expected = np.concatenate([e, -e]).T
    np.testing.assert_array_equal(uc.im2col(inputs, filters_shape), expected)


def test_col2im():
    inputs_shape = (2, 3, 3, 3)
    filters_shape = (1, 3, 2, 2)
    partial_cols = np.arange(1, 49).reshape(4, 12).T
    cols = np.hstack([partial_cols, -partial_cols])
    partial_expected = np.array(
        [
            [[1, 15, 14], [28, 82, 54], [27, 67, 40]],
            [[5, 23, 18], [36, 98, 62], [31, 75, 44]],
            [[9, 31, 22], [44, 114, 70], [35, 83, 48]],
        ]
    )
    expected = np.stack([partial_expected, -partial_expected])
    np.testing.assert_array_equal(
        expected, uc.col2im(cols, inputs_shape, filters_shape)
    )


def test_col2im_padding():
    inputs_shape = (2, 2, 2, 2)
    filters_shape = (1, 2, 2, 2)
    cols = np.ones(shape=(2 * 2 * 2, 9 * 2), dtype=np.int64)
    expected = np.ones(shape=(2, 2, 2, 2)) * 4
    np.testing.assert_array_equal(
        expected, uc.col2im(cols, inputs_shape, filters_shape, padding=1)
    )


def test_col2im_stride():
    inputs_shape = (1, 1, 5, 5)
    filters_shape = (1, 1, 3, 3)
    cols = np.ones(shape=(9, 4), dtype=np.int64)
    expected = np.array(
        [
            [1, 1, 2, 1, 1],
            [1, 1, 2, 1, 1],
            [2, 2, 4, 2, 2],
            [1, 1, 2, 1, 1],
            [1, 1, 2, 1, 1],
        ]
    ).reshape(1, 1, 5, 5)
    actual = uc.col2im(cols, inputs_shape, filters_shape, stride=2)
    np.testing.assert_array_equal(expected, actual)


def test_convert_filters_back_and_forth():
    f = np.random.randint(1, 10, (2, 3, 2, 2))
    fc = uc.im2col_filters(f)
    f_back = uc.col2im_filters(fc, f.shape)
    np.testing.assert_array_equal(f, f_back)
