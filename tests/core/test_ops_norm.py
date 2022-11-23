import numpy as np
import pytest
from numpy import array
from numpy.testing import assert_allclose

from chains.core.ops_norm import BatchNormTraining, BatchNormPredict
from chains.core.static_shape import StaticShape, Dim


def test_batch_norm_computation_and_gradient():
    np.random.seed(1)
    x = np.random.rand(2, 3).astype("float32")
    d_out = np.random.rand(2, 3).astype("float32")
    bn = BatchNormTraining(momentum=0.99, epsilon=1e-3)
    beta = np.zeros((2, 1), dtype="float32")
    gamma = np.ones((2, 1), dtype="float32")
    beta_shape = StaticShape(2, 1)
    gamma_shape = StaticShape(2, 1)
    x_shape = StaticShape(2, Dim.unknown())
    bn.check_incoming_shapes(beta_shape, gamma_shape, x_shape)

    assert bn.compute_out_shape(beta_shape, gamma_shape, x_shape) == x_shape

    bn.compute(beta, gamma, x)
    assert_allclose(
        bn.output,
        array([[0.12753308, 1.1489942, -1.276527], [1.29037, -0.35706627, -0.9333031]]),
        atol=1e-7,
    )

    d_beta, d_gamma, d_x = bn.partials(d_out)
    assert_allclose(d_beta, array([[0.9285884], [1.6432308]]))
    assert_allclose(d_gamma, array([[-0.08568269], [-0.09392489]]), atol=1e-6)
    assert_allclose(
        d_x,
        array(
            [[-0.40287876, 0.23186469, 0.17101419], [0.33326817, -1.4796133, 1.1463442]]
        ),
        atol=1e-6,
    )


def test_batch_norm_moving_avg():
    np.random.seed(1)
    x = np.random.rand(2, 3)

    bn = BatchNormTraining(momentum=0.9, epsilon=1e-3, sample_axis=0)
    beta = np.zeros((2, 1))
    gamma = np.ones((2, 1))

    for _ in range(400):
        bn.compute(beta, gamma, x)

    assert_allclose(bn.avg, np.mean(x, axis=0, keepdims=True))
    assert_allclose(bn.var, np.var(x, axis=0, keepdims=True))


def test_batch_norm_predict():
    np.random.seed(1)
    x = np.random.rand(2, 3)

    bn = BatchNormTraining(momentum=0.9, epsilon=1e-3, sample_axis=0)
    bn.avg = 5.0
    bn.var = 1.0
    bnp = BatchNormPredict(bn)
    bnd = BatchNormPredict(avg=5.0, var=1.0, epsilon=1e-3)
    beta = np.zeros((2, 1))
    gamma = np.ones((2, 1))

    assert bnp.compute(beta, gamma, x) == bn.compute(beta, gamma, x)
    assert bnd.compute(beta, gamma, x) == bn.compute(beta, gamma, x)


def test_batch_norm_incorrect_shape():
    beta_shape = StaticShape(2, 1)
    gamma_shape = StaticShape(1, 2)
    x_shape = StaticShape(2, Dim.unknown())
    bn = BatchNormTraining(momentum=0.99, epsilon=1e-3)

    with pytest.raises(ValueError):
        bn.check_incoming_shapes(beta_shape, gamma_shape, x_shape)


def _sample_batch_norm_case():
    import tensorflow as tf

    np.random.seed(1)
    initial = np.random.rand(2, 3).astype("float32")
    loss = tf.constant(np.random.rand(2, 3).astype("float32"))
    x = tf.Variable(initial, trainable=True)
    x_norm = tf.layers.batch_normalization(
        inputs=x, axis=0, momentum=0.99, epsilon=1e-3, training=True
    )

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    grads = tf.gradients(ys=x_norm, xs=vars, grad_ys=loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("x_norm  = ", sess.run(x_norm))
        # g, ex = sess.run([grads_and_vars, extra_update_ops])
        # print("grads   = ", len(g[0]))
        # print("moving_averages   = ", ex)

        for euo in extra_update_ops:
            print(euo)

        for v, g in zip(vars, sess.run(grads)):
            print(v, " = ", g)
