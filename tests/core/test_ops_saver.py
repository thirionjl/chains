import numpy as np

from chains.core import saver
from chains.core.initializers import ConstantInitializer
from chains.core.ops import Var
from chains.core.ops_norm import BatchNormTraining
from chains.core.static_shape import StaticShape


def test_save_var_back_and_forth():
    to_save = np.array(50.0)
    var = Var(initializer=ConstantInitializer(to_save),
              shape=StaticShape.from_tuple(to_save.shape),
              dtype=to_save.dtype)
    var.initialize_if_needed()
    bak = saver.save(var)
    var.output = np.array(0.0)
    assert var.output != to_save

    saver.restore(var, bak)
    assert var.output == to_save


def test_save_var_back_and_forth2():
    to_save = 50.0
    var = Var(initializer=ConstantInitializer(to_save),
              shape=StaticShape.from_tuple(()),
              dtype='float')
    var.initialize_if_needed()
    bak = saver.save(var)
    var.output = np.array(0.0)
    assert var.output != to_save

    saver.restore(var, bak)
    assert var.output == to_save


def test_save_batch_norm_back_and_forth():
    bn = BatchNormTraining(momentum=0.91, epsilon=1e-4, sample_axis=-1)
    bn.var = np.array(80.0)
    bn.avg = np.array(18.0)

    bak = saver.save(bn)
    bn.var = None
    bn.avg = None

    saver.restore(bn, bak)
    assert bn.var == np.array(80.0)
    assert bn.avg == np.array(18.0)
