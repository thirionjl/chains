import numpy as np

from chains.core import ops_losses as losses


def test_sigmoid_cross_entropy_with_logits():
    sce_logits = losses.SigmoidCrossEntropyWithLogits()
    labels = np.array([0, 1])
    logits = np.array([0, 2])

    sce_logits.compute(logits, labels)

    np.testing.assert_allclose(sce_logits.activations,
                               np.array([0.5, 0.88079708]))
    np.testing.assert_allclose(sce_logits.output,
                               np.array([0.69314718, 0.12692801]))
    np.testing.assert_allclose(sce_logits.partials(1)[0],
                               np.array([0.5, -0.11920292]))


def test_sigmoid_cross_entropy():
    sce = losses.SigmoidCrossEntropy()
    labels = np.array([[0.0, 1.0]])
    logits = np.array([[0.0, 2.0]])

    sce.compute(logits, labels)

    np.testing.assert_allclose(sce.activations, np.array([[0.5, 0.88079708]]))
    np.testing.assert_allclose(sce.output, 0.4100375958014589)
    np.testing.assert_allclose(sce.partials(1)[0],
                               np.array([[0.25, -0.05960146]]))


def test_softmax_cross_entropy_with_logits():
    sce_logits = losses.SoftMaxCrossEntropyWithLogits(class_axis=0, epsilon=0)
    labels = np.array([[1, 0],
                       [0, 0],
                       [0, 0],
                       [0, 1]])
    logits = np.array([[0., 1.],
                       [2., 3.],
                       [4., 5.],
                       [6., 17.]])

    sce_logits.compute(logits, labels)

    np.testing.assert_allclose(sce_logits.activations,
                               np.array([[2.14400878e-03, 1.12534377e-07],
                                         [1.58422012e-02, 8.31522825e-07],
                                         [1.17058913e-01, 6.14416880e-06],
                                         [8.64954877e-01, 9.99992912e-01]]))
    np.testing.assert_allclose(sce_logits.output,
                               np.array([6.14507794e+00, 7.08825113e-06]))
    np.testing.assert_allclose(sce_logits.partials(1)[0],
                               np.array([[-9.97855991e-01, 1.12534377e-07],
                                         [1.58422012e-02, 8.31522825e-07],
                                         [1.17058913e-01, 6.14416880e-06],
                                         [8.64954877e-01, -7.08822600e-06]]))


def test_softmax_cross_entropy():
    sce = losses.SoftMaxCrossEntropy(class_axis=0)
    labels = np.array([[1, 0],
                       [0, 0],
                       [0, 0],
                       [0, 1]])
    logits = np.array([[0., 1.],
                       [2., 3.],
                       [4., 5.],
                       [6., 17.]])

    sce.compute(logits, labels)

    np.testing.assert_allclose(sce.activations,
                               np.array([[2.14400878e-03, 1.12534377e-07],
                                         [1.58422012e-02, 8.31522825e-07],
                                         [1.17058913e-01, 6.14416880e-06],
                                         [8.64954877e-01, 9.99992912e-01]]))
    np.testing.assert_allclose(sce.output, 3.072542513605954)
    np.testing.assert_allclose(sce.partials(1)[0],
                               np.array([[-4.98927996e-01, 5.62671885e-08],
                                         [7.92110059e-03, 4.15761413e-07],
                                         [5.85294566e-02, 3.07208440e-06],
                                         [4.32477438e-01, -3.54411300e-06]])
                               )


def _sample_sigmoid_case():
    import tensorflow as tf
    labels = tf.constant(np.array([0.0, 1.0]))
    logits = tf.Variable(np.array([0.0, 2.0]))

    sigmoid_ce_with_logits = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    sigmoid_ce = tf.reduce_mean(sigmoid_ce_with_logits)
    sigmoid = tf.nn.sigmoid(logits)
    gd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    grads_and_vars = gd.compute_gradients(sigmoid_ce)
    grads_and_vars_l = gd.compute_gradients(sigmoid_ce_with_logits)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ce = sess.run(sigmoid_ce)
        ce_logits = sess.run(sigmoid_ce_with_logits)
        s = sess.run(sigmoid)
        g = sess.run(grads_and_vars)
        g_l = sess.run(grads_and_vars_l)
        print("Activations  = ", s)
        print("Cross Entropy = ", ce)
        print("Cross Entropy Logits = ", ce_logits)
        print("d   = ", g[0][0])
        print("d (logits) = ", g_l[0][0])


def _sample_softmax_case():
    import tensorflow as tf
    la = np.array([[1, 0],
                   [0, 0],
                   [0, 0],
                   [0, 1]])
    lo = np.array([[0., 1.],
                   [2., 3.],
                   [4., 5.],
                   [6., 17.]])
    labels = tf.constant(la.T)
    logits = tf.Variable(lo.T)

    softmax_ce_with_logits = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits)
    softmax_ce = tf.reduce_mean(softmax_ce_with_logits)
    softmax = tf.nn.softmax(logits)
    gd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    grads_and_vars_l = gd.compute_gradients(softmax_ce_with_logits)
    grads_and_vars = gd.compute_gradients(softmax_ce)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ce = sess.run(softmax_ce)
        ce_logits = sess.run(softmax_ce_with_logits)
        s = sess.run(softmax)
        g_l = sess.run(grads_and_vars_l)
        g = sess.run(grads_and_vars)
        print("Activations  = ", s)
        print("Cross Entropy = ", ce)
        print("Cross Entropy Logits = ", ce_logits)
        print("d = ", g[0][0])
        print("d (logits) = ", g_l[0][0])
