from numpy import array
from numpy.testing import assert_allclose

from chains.core.graph import Graph
from chains.core.node_factory import initialized_var
from chains.core.optimizers import MomentumOptimizer, AdamOptimizer
from chains.core.static_shape import StaticShape


class DummyGraph(Graph):

    def __init__(self):
        self.w1 = initialized_var("W1", array(
            [[1.62434536, -0.61175641, -0.52817175],
             [-1.07296862, 0.86540763, -2.3015387]]))
        self.b1 = initialized_var("b1", array([[1.74481176], [-0.7612069]]))
        self.w2 = initialized_var("W2", array(
            [[0.3190391, -0.24937038, 1.46210794],
             [-2.06014071, -0.3224172, -0.38405435],
             [1.13376944, -1.09989127, -0.17242821]]))
        self.b2 = initialized_var("b2", array(
            [[-0.87785842], [0.04221375], [0.58281521]]))

        self.variables = {self.w1, self.b1, self.w2, self.b2}
        self._placeholders = {}

    def forward(self):
        return 0

    def backward(self):
        return {
            self.w1: array([[-1.10061918, 1.14472371, 0.90159072],
                            [0.50249434, 0.90085595, -0.68372786]]),
            self.b1: array([[-0.12289023], [-0.93576943]]),
            self.w2: array([[-0.26788808, 0.53035547, -0.69166075],
                            [-0.39675353, -0.6871727, -0.84520564],
                            [-0.67124613, -0.0126646, -1.11731035]]),
            self.b2: array([[0.2344157], [1.65980218], [0.74204416]])
        }

    @property
    def shape(self):
        return StaticShape.scalar()


def test_momentum_coursera():
    g = DummyGraph()
    g.initialize_variables()

    optimizer = MomentumOptimizer(lr=0.01, beta=0.9)
    optimizer.prepare_and_check(g)

    optimizer.run()

    assert_allclose(g.w1.value, array(
        [[1.62544598, -0.61290114, -0.52907334],
         [-1.07347112, 0.86450677, -2.30085497]]))

    assert_allclose(g.b1.value,
                    array([[1.74493465], [-0.76027113]]))

    assert_allclose(g.w2.value,
                    array([[0.31930698, -0.24990073, 1.4627996],
                           [-2.05974396, -0.32173003,
                            -0.38320915],
                           [1.13444069, -1.0998786,
                            -0.1713109]]))

    assert_allclose(g.b2.value, array(
        [[-0.87809283], [0.04055394], [0.58207317]]), atol=1e-8)

    assert_allclose(optimizer.v[g.w1], array(
        [[-0.11006192, 0.11447237, 0.09015907],
         [0.05024943, 0.09008559, -0.06837279]]))

    assert_allclose(optimizer.v[g.b1], array([[-0.01228902], [-0.09357694]]),
                    atol=1e-8)
    assert_allclose(optimizer.v[g.w2], array(
        [[-0.02678881, 0.05303555, -0.06916608],
         [-0.03967535, -0.06871727, -0.08452056],
         [-0.06712461, -0.00126646, -0.11173103]]))

    assert_allclose(optimizer.v[g.b2],
                    array([[0.02344157], [0.16598022], [0.07420442]]))


def test_adam_coursera():
    g = DummyGraph()
    g.initialize_variables()

    optimizer = AdamOptimizer(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    optimizer.prepare_and_check(g)

    # Simulate second mini-batch as in coursera test case
    optimizer.t = 1
    optimizer.beta1_pow = optimizer.beta1
    optimizer.beta2_pow = optimizer.beta2

    optimizer.run()

    # Verify variables values
    assert_allclose(g.w1.value, array([[1.63178673, -0.61919778, -0.53561312],
                                       [-1.08040999, 0.85796626,
                                        -2.29409733]]))
    assert_allclose(g.b1.value, array([[1.75225313], [-0.75376553]]))
    assert_allclose(g.w2.value, array([[0.32648046, -0.25681174, 1.46954931],
                                       [-2.05269934, -0.31497584, -0.37661299],
                                       [1.14121081, -1.09244991,
                                        -0.16498684]]), atol=1e-6)
    assert_allclose(g.b2.value,
                    array([[-0.88529979], [0.03477238], [0.57537385]]))

    # Verify velocities
    assert_allclose(optimizer.v[g.w1], array(
        [[-0.11006192, 0.11447237, 0.09015907],
         [0.05024943, 0.09008559, -0.06837279]]))
    assert_allclose(optimizer.v[g.b1], array([[-0.01228902], [-0.09357694]]),
                    atol=1e-8)
    assert_allclose(optimizer.v[g.w2], array(
        [[-0.02678881, 0.05303555, -0.06916608],
         [-0.03967535, -0.06871727, -0.08452056],
         [-0.06712461, -0.00126646, -0.11173103]]))
    assert_allclose(optimizer.v[g.b2],
                    array([[0.02344157], [0.16598022], [0.07420442]]))

    # Verify RMS
    assert_allclose(optimizer.s[g.w1], array(
        [[0.00121136, 0.00131039, 0.00081287],
         [0.0002525, 0.00081154, 0.00046748]]), atol=1e-8)
    assert_allclose(optimizer.s[g.b1],
                    array([[1.51020075e-05], [8.75664434e-04]]))
    assert_allclose(optimizer.s[g.w2], array(
        [[7.17640232e-05, 2.81276921e-04, 4.78394595e-04],
         [1.57413361e-04, 4.72206320e-04, 7.14372576e-04],
         [4.50571368e-04, 1.60392066e-07, 1.24838242e-03]]), atol=1e-8)
    assert_allclose(optimizer.s[g.b2], array(
        [[5.49507194e-05], [2.75494327e-03], [5.50629536e-04]]))

    # Verify beta1 and beta2 powers
    assert_allclose(optimizer.beta1_pow, optimizer.beta1 ** 2)
    assert_allclose(optimizer.beta2_pow, optimizer.beta2 ** 2)
