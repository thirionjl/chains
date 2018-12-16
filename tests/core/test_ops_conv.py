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

    assert str(ex.value) == f"Number of channels should be the same " \
        f"in features(3) and filters(4)"


def test_invalid_bias():
    conv = Conv2D()
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)
    filters = StaticShape(16, 3, 2, 2)
    biases = StaticShape(17, 1)

    with pytest.raises(ValueError) as ex:
        conv.check_incoming_shapes(features, filters, biases)
    assert str(ex.value) == f"Number of bias should match number of filters " \
        f"but got 17 and 16"


def test_compute_valid_conv():
    conv = Conv2D(conv_format=TensorFlowNHWC)

    features = np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3)
    filters = np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2)
    biases = np.zeros((2, 1))

    conv.check_incoming_shapes(StaticShape.of_tensor(features),
                               StaticShape.of_tensor(filters),
                               StaticShape.of_tensor(biases))
    conv.compute(features, filters, biases)
    actual = conv.output
    assert actual.shape == (2, 2, 3, 2)

    np.testing.assert_allclose(actual, np.array([[[[1624., 1726.],
                                                   [2020., 2158.],
                                                   [2416., 2590.]],

                                                  [[3208., 3454.],
                                                   [3604., 3886.],
                                                   [4000., 4318.]]],

                                                 [[[6376., 6910.],
                                                   [6772., 7342.],
                                                   [7168., 7774.]],

                                                  [[7960., 8638.],
                                                   [8356., 9070.],
                                                   [8752., 9502.]]]],
                                                dtype=np.float32))

    dfeatures, dfilters, dbias = conv.partials(np.ones(actual.shape))

    assert dfeatures.shape == features.shape
    assert dfilters.shape == filters.shape
    assert dbias.shape == biases.shape

    np.testing.assert_allclose(dfeatures, np.array([[[[1., 5., 9.],
                                                      [14., 22., 30.],
                                                      [14., 22., 30.],
                                                      [13., 17., 21.]],

                                                     [[26., 34., 42.],
                                                      [76., 92., 108.],
                                                      [76., 92., 108.],
                                                      [50., 58., 66.]],

                                                     [[25., 29., 33.],
                                                      [62., 70., 78.],
                                                      [62., 70., 78.],
                                                      [37., 41., 45.]]],

                                                    [[[1., 5., 9.],
                                                      [14., 22., 30.],
                                                      [14., 22., 30.],
                                                      [13., 17., 21.]],

                                                     [[26., 34., 42.],
                                                      [76., 92., 108.],
                                                      [76., 92., 108.],
                                                      [50., 58., 66.]],

                                                     [[25., 29., 33.],
                                                      [62., 70., 78.],
                                                      [62., 70., 78.],
                                                      [37., 41., 45.]]]],
                                                   dtype=np.float32))

    np.testing.assert_allclose(dfilters, np.array([[[[324., 324.],
                                                     [336., 336.],
                                                     [348., 348.]],

                                                    [[360., 360.],
                                                     [372., 372.],
                                                     [384., 384.]]],

                                                   [[[468., 468.],
                                                     [480., 480.],
                                                     [492., 492.]],

                                                    [[504., 504.],
                                                     [516., 516.],
                                                     [528., 528.]]]],
                                                  dtype=np.float32))


def test_compute_same_conv():
    conv = Conv2D(conv_format=TensorFlowNHWC, padding=1)

    features = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    filters = np.arange(9, dtype=np.float32).reshape(3, 3, 1, 1)
    biases = np.zeros((1, 1))

    conv.check_incoming_shapes(StaticShape.of_tensor(features),
                               StaticShape.of_tensor(filters),
                               StaticShape.of_tensor(biases))
    conv.compute(features, filters, biases)
    actual = conv.output
    assert actual.shape == (1, 4, 4, 1)

    np.testing.assert_allclose(actual, np.array([[[[73.],
                                                   [121.],
                                                   [154.],
                                                   [103.]],

                                                  [[171.],
                                                   [258.],
                                                   [294.],
                                                   [186.]],

                                                  [[279.],
                                                   [402.],
                                                   [438.],
                                                   [270.]],

                                                  [[139.],
                                                   [187.],
                                                   [202.],
                                                   [113.]]]],
                                                dtype=np.float32))

    dx, df, db = conv.partials(np.ones(actual.shape))

    assert dx.shape == features.shape
    assert df.shape == filters.shape
    assert db.shape == biases.shape

    np.testing.assert_allclose(dx, np.array([[[[8.],
                                               [15.],
                                               [15.],
                                               [12.]],

                                              [[21.],
                                               [36.],
                                               [36.],
                                               [27.]],

                                              [[21.],
                                               [36.],
                                               [36.],
                                               [27.]],

                                              [[20.],
                                               [33.],
                                               [33.],
                                               [24.]]]],
                                            dtype=np.float32))

    np.testing.assert_allclose(df, np.array([[[[45.]],

                                              [[66.]],

                                              [[54.]]],

                                             [[[84.]],

                                              [[120.]],

                                              [[96.]]],

                                             [[[81.]],

                                              [[114.]],

                                              [[90.]]]],
                                            dtype=np.float32))


def test_compute_strided_conv():
    conv = Conv2D(conv_format=TensorFlowNHWC, stride=2)

    features = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    filters = np.arange(4, dtype=np.float32).reshape(2, 2, 1, 1)
    biases = np.zeros((1, 1))

    conv.check_incoming_shapes(StaticShape.of_tensor(features),
                               StaticShape.of_tensor(filters),
                               StaticShape.of_tensor(biases))
    conv.compute(features, filters, biases)
    actual = conv.output
    assert actual.shape == (1, 2, 2, 1)

    np.testing.assert_allclose(actual, np.array([[[[24.],
                                                   [36.]],

                                                  [[72.],
                                                   [84.]]]], dtype=np.float32))

    dx, df, db = conv.partials(np.ones(actual.shape))

    assert dx.shape == features.shape
    assert df.shape == filters.shape
    assert db.shape == biases.shape

    np.testing.assert_allclose(dx, np.array([[[[0.],
                                               [1.],
                                               [0.],
                                               [1.]],

                                              [[2.],
                                               [3.],
                                               [2.],
                                               [3.]],

                                              [[0.],
                                               [1.],
                                               [0.],
                                               [1.]],

                                              [[2.],
                                               [3.],
                                               [2.],
                                               [3.]]]], dtype=np.float32))

    np.testing.assert_allclose(df, np.array([[[[20.]],

                                              [[24.]]],

                                             [[[36.]],

                                              [[40.]]]], dtype=np.float32))


def _sample_conv2d_case_no_stride():
    import tensorflow as tf

    features = tf.Variable(np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3))
    filters = tf.Variable(np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2))

    conv = tf.nn.conv2d(input=features, filter=filters, padding="VALID",
                        strides=(1, 1, 1, 1))
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

    conv = tf.nn.conv2d(input=features, filter=filters, padding="VALID",
                        strides=(1, 2, 2, 1))
    gd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    grads_and_vars = gd.compute_gradients(conv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c = sess.run(conv)
        g = sess.run(grads_and_vars)
        print("Activations  = ", repr(c))
        print("dFeatures  = ", repr(g[0][0]))
        print("dFilters  = ", repr(g[1][0]))
