import numpy as np
import pytest

from chains.core.ops_conv import Conv2D
from chains.core.static_shape import StaticShape, Dim
from chains.core.utils_conv import TensorFlowNHWC


def test_valid_shape_without_padding_and_stride():
    conv = Conv2D(padding=0, stride=1)
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)
    filters = StaticShape(16, 3, 9, 9)
    biases = StaticShape(16, 1)

    conv.check_incoming_shapes(features, filters, biases)
    out_shape = conv.compute_out_shape(features, filters, biases)
    assert out_shape == StaticShape(m, 16, 32, 22)


def test_valid_shape_with_padding():
    conv = Conv2D(padding=4, stride=1)
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)
    filters = StaticShape(16, 3, 9, 9)
    biases = StaticShape(16, 1)

    conv.check_incoming_shapes(features, filters, biases)
    out_shape = conv.compute_out_shape(features, filters, biases)
    assert out_shape == StaticShape(m, 16, 40, 30)


def test_valid_shape_with_padding_and_stride():
    conv = Conv2D(padding=1, stride=2)
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)
    filters = StaticShape(16, 3, 2, 2)
    biases = StaticShape(16, 1)

    conv.check_incoming_shapes(features, filters, biases)
    out_shape = conv.compute_out_shape(features, filters, biases)
    assert out_shape == StaticShape(m, 16, 21, 16)


def test_invalid_number_of_channels():
    conv = Conv2D()
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)
    filters = StaticShape(16, 4, 2, 2)
    biases = StaticShape(16, 1)

    with pytest.raises(ValueError) as ex:
        conv.check_incoming_shapes(features, filters, biases)

    assert (
        str(ex.value)
        == "Number of channels should be the same in features(3) and filters(4)"
    )


def test_invalid_bias():
    conv = Conv2D()
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)
    filters = StaticShape(16, 3, 2, 2)
    biases = StaticShape(17, 1)

    with pytest.raises(ValueError) as ex:
        conv.check_incoming_shapes(features, filters, biases)
    assert (
        str(ex.value)
        == "Number of bias should match number of filters but got 17 and 16"
    )


def test_compute_valid_conv():
    conv = Conv2D(conv_format=TensorFlowNHWC)

    features = np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3)
    filters = np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2)
    biases = np.zeros((2, 1))

    conv.check_incoming_shapes(
        StaticShape.of_tensor(features),
        StaticShape.of_tensor(filters),
        StaticShape.of_tensor(biases),
    )
    conv.compute(features, filters, biases)
    actual = conv.output
    assert actual.shape == (2, 2, 3, 2)

    np.testing.assert_allclose(
        actual,
        np.array(
            [
                [
                    [[1624.0, 1726.0], [2020.0, 2158.0], [2416.0, 2590.0]],
                    [[3208.0, 3454.0], [3604.0, 3886.0], [4000.0, 4318.0]],
                ],
                [
                    [[6376.0, 6910.0], [6772.0, 7342.0], [7168.0, 7774.0]],
                    [[7960.0, 8638.0], [8356.0, 9070.0], [8752.0, 9502.0]],
                ],
            ],
            dtype=np.float32,
        ),
    )

    dfeatures, dfilters, dbias = conv.partials(np.ones(actual.shape))

    assert dfeatures.shape == features.shape
    assert dfilters.shape == filters.shape
    assert dbias.shape == biases.shape

    np.testing.assert_allclose(
        dfeatures,
        np.array(
            [
                [
                    [
                        [1.0, 5.0, 9.0],
                        [14.0, 22.0, 30.0],
                        [14.0, 22.0, 30.0],
                        [13.0, 17.0, 21.0],
                    ],
                    [
                        [26.0, 34.0, 42.0],
                        [76.0, 92.0, 108.0],
                        [76.0, 92.0, 108.0],
                        [50.0, 58.0, 66.0],
                    ],
                    [
                        [25.0, 29.0, 33.0],
                        [62.0, 70.0, 78.0],
                        [62.0, 70.0, 78.0],
                        [37.0, 41.0, 45.0],
                    ],
                ],
                [
                    [
                        [1.0, 5.0, 9.0],
                        [14.0, 22.0, 30.0],
                        [14.0, 22.0, 30.0],
                        [13.0, 17.0, 21.0],
                    ],
                    [
                        [26.0, 34.0, 42.0],
                        [76.0, 92.0, 108.0],
                        [76.0, 92.0, 108.0],
                        [50.0, 58.0, 66.0],
                    ],
                    [
                        [25.0, 29.0, 33.0],
                        [62.0, 70.0, 78.0],
                        [62.0, 70.0, 78.0],
                        [37.0, 41.0, 45.0],
                    ],
                ],
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        dfilters,
        np.array(
            [
                [
                    [[324.0, 324.0], [336.0, 336.0], [348.0, 348.0]],
                    [[360.0, 360.0], [372.0, 372.0], [384.0, 384.0]],
                ],
                [
                    [[468.0, 468.0], [480.0, 480.0], [492.0, 492.0]],
                    [[504.0, 504.0], [516.0, 516.0], [528.0, 528.0]],
                ],
            ],
            dtype=np.float32,
        ),
    )


def test_compute_same_conv():
    conv = Conv2D(conv_format=TensorFlowNHWC, padding=1)

    features = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    filters = np.arange(9, dtype=np.float32).reshape(3, 3, 1, 1)
    biases = np.zeros((1, 1))

    conv.check_incoming_shapes(
        StaticShape.of_tensor(features),
        StaticShape.of_tensor(filters),
        StaticShape.of_tensor(biases),
    )
    conv.compute(features, filters, biases)
    actual = conv.output
    assert actual.shape == (1, 4, 4, 1)

    np.testing.assert_allclose(
        actual,
        np.array(
            [
                [
                    [[73.0], [121.0], [154.0], [103.0]],
                    [[171.0], [258.0], [294.0], [186.0]],
                    [[279.0], [402.0], [438.0], [270.0]],
                    [[139.0], [187.0], [202.0], [113.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )

    dx, df, db = conv.partials(np.ones(actual.shape))

    assert dx.shape == features.shape
    assert df.shape == filters.shape
    assert db.shape == biases.shape

    np.testing.assert_allclose(
        dx,
        np.array(
            [
                [
                    [[8.0], [15.0], [15.0], [12.0]],
                    [[21.0], [36.0], [36.0], [27.0]],
                    [[21.0], [36.0], [36.0], [27.0]],
                    [[20.0], [33.0], [33.0], [24.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        df,
        np.array(
            [
                [[[45.0]], [[66.0]], [[54.0]]],
                [[[84.0]], [[120.0]], [[96.0]]],
                [[[81.0]], [[114.0]], [[90.0]]],
            ],
            dtype=np.float32,
        ),
    )


def test_compute_strided_conv():
    conv = Conv2D(conv_format=TensorFlowNHWC, stride=2)

    features = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    filters = np.arange(4, dtype=np.float32).reshape(2, 2, 1, 1)
    biases = np.zeros((1, 1))

    conv.check_incoming_shapes(
        StaticShape.of_tensor(features),
        StaticShape.of_tensor(filters),
        StaticShape.of_tensor(biases),
    )
    conv.compute(features, filters, biases)
    actual = conv.output
    assert actual.shape == (1, 2, 2, 1)

    np.testing.assert_allclose(
        actual, np.array([[[[24.0], [36.0]], [[72.0], [84.0]]]], dtype=np.float32)
    )

    dx, df, db = conv.partials(np.ones(actual.shape))

    assert dx.shape == features.shape
    assert df.shape == filters.shape
    assert db.shape == biases.shape

    np.testing.assert_allclose(
        dx,
        np.array(
            [
                [
                    [[0.0], [1.0], [0.0], [1.0]],
                    [[2.0], [3.0], [2.0], [3.0]],
                    [[0.0], [1.0], [0.0], [1.0]],
                    [[2.0], [3.0], [2.0], [3.0]],
                ]
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_allclose(
        df, np.array([[[[20.0]], [[24.0]]], [[[36.0]], [[40.0]]]], dtype=np.float32)
    )


def _sample_conv2d_case_no_stride():
    import tensorflow as tf

    features = tf.Variable(np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3))
    filters = tf.Variable(np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2))

    conv = tf.nn.conv2d(
        input=features, filter=filters, padding="VALID", strides=(1, 1, 1, 1)
    )
    gd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    grads_and_vars = gd.compute_gradients(conv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c = sess.run(conv)
        g = sess.run(grads_and_vars)
        print("Activations  = ", repr(c))
        print("dFeatures  = ", repr(g[0][0]))
        print("dFilters  = ", repr(g[1][0]))


def _sample_conv2d_strided():
    import tensorflow as tf

    features = tf.Variable(np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1))
    filters = tf.Variable(np.arange(4, dtype=np.float32).reshape(2, 2, 1, 1))

    conv = tf.nn.conv2d(
        input=features, filter=filters, padding="VALID", strides=(1, 2, 2, 1)
    )
    gd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    grads_and_vars = gd.compute_gradients(conv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c = sess.run(conv)
        g = sess.run(grads_and_vars)
        print("Activations  = ", repr(c))
        print("dFeatures  = ", repr(g[0][0]))
        print("dFilters  = ", repr(g[1][0]))
