import numpy as np
import pytest

from chains.core.ops_pooling import MaxPool
from chains.core.static_shape import StaticShape, Dim
from chains.core.utils_conv import TensorFlowNHWC


def test_valid_shape_and_stride():
    pool = MaxPool(stride=2)
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)

    pool.check_incoming_shapes(features)
    out_shape = pool.compute_out_shape(features)
    assert out_shape == StaticShape(m, 3, 20, 15)


def test_invalid_shape_and_stride():
    pool = MaxPool(stride=3)
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)

    with pytest.raises(ValueError) as ex:
        pool.check_incoming_shapes(features)
    assert str(
        ex.value) == "Height (40) should be a multiple of stride 3 but is not"


def test_compute_max_pool():
    pool = MaxPool(stride=2)

    features = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)

    pool.check_incoming_shapes(StaticShape.of_tensor(features))
    pool.compute(features)
    actual = pool.output
    assert actual.shape == (2, 3, 2, 2)

    e1 = np.array([5, 7, 13, 15, 21, 23, 29, 31, 37, 39, 45, 47]) \
        .reshape(3, 2, 2).astype(np.float32)
    expected = np.stack([e1, e1 + 48], axis=0)
    assert expected.shape == (2, 3, 2, 2)

    np.testing.assert_allclose(actual, expected)

    d_features, = pool.partials(np.ones(actual.shape))
    assert d_features.shape == features.shape

    row1 = [0, 0, 0, 0]
    row2 = [0, 1, 0, 1]
    channel = np.stack([row1, row2, row1, row2], axis=0)
    sample = np.stack([channel, channel, channel], axis=0)
    expected_partial = np.stack([sample, sample], axis=0)

    np.testing.assert_allclose(d_features, expected_partial)


def test_compute_max_pool_with_other_format():
    pool = MaxPool(stride=2, conv_format=TensorFlowNHWC)

    features = np.arange(2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)

    pool.check_incoming_shapes(StaticShape.of_tensor(features))
    pool.compute(features)
    actual = pool.output

    np.testing.assert_allclose(actual, np.array([[[[15., 16., 17.],
                                                   [21., 22., 23.]],

                                                  [[39., 40., 41.],
                                                   [45., 46., 47.]]],

                                                 [[[63., 64., 65.],
                                                   [69., 70., 71.]],

                                                  [[87., 88., 89.],
                                                   [93., 94., 95.]]]],
                                                dtype=np.float32))

    d_features, = pool.partials(np.ones(actual.shape))

    np.testing.assert_allclose(d_features, np.array([[[[0., 0., 0.],
                                                       [0., 0., 0.],
                                                       [0., 0., 0.],
                                                       [0., 0., 0.]],

                                                      [[0., 0., 0.],
                                                       [1., 1., 1.],
                                                       [0., 0., 0.],
                                                       [1., 1., 1.]],

                                                      [[0., 0., 0.],
                                                       [0., 0., 0.],
                                                       [0., 0., 0.],
                                                       [0., 0., 0.]],

                                                      [[0., 0., 0.],
                                                       [1., 1., 1.],
                                                       [0., 0., 0.],
                                                       [1., 1., 1.]]],

                                                     [[[0., 0., 0.],
                                                       [0., 0., 0.],
                                                       [0., 0., 0.],
                                                       [0., 0., 0.]],

                                                      [[0., 0., 0.],
                                                       [1., 1., 1.],
                                                       [0., 0., 0.],
                                                       [1., 1., 1.]],

                                                      [[0., 0., 0.],
                                                       [0., 0., 0.],
                                                       [0., 0., 0.],
                                                       [0., 0., 0.]],

                                                      [[0., 0., 0.],
                                                       [1., 1., 1.],
                                                       [0., 0., 0.],
                                                       [1., 1., 1.]]]],
                                                    dtype=np.float32))


def _sample_pool2d_strided():
    import tensorflow as tf

    features = tf.Variable(
        np.arange(2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3))

    conv = tf.nn.max_pool(value=features, ksize=(1, 2, 2, 1), padding="VALID",
                          strides=(1, 2, 2, 1))
    gd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    grads_and_vars = gd.compute_gradients(conv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c = sess.run(conv)
        g = sess.run(grads_and_vars)
        print("Activations  = ", repr(c))
        print("dFeatures  = ", repr(g[0][0]))
