import numpy as np
import pytest

from chains.core.ops_mat import ArgMax
from chains.core.static_shape import StaticShape


def test_argmax():
    a = np.arange(6).reshape(2, 3)
    op1 = ArgMax(axis=0)
    op1.compute(a)
    op2 = ArgMax(axis=1)
    op2.compute(a)

    np.testing.assert_equal(op1.output, np.array([1, 1, 1]))
    np.testing.assert_equal(op2.output, np.array([2, 2]))

    op1.check_incoming_shapes(StaticShape.from_tuple(a.shape))
    op2.check_incoming_shapes(StaticShape.from_tuple(a.shape))

    out_shape1 = op1.compute_out_shape(StaticShape.from_tuple(a.shape))
    out_shape2 = op2.compute_out_shape(StaticShape.from_tuple(a.shape))
    assert out_shape1.to_numpy() == (3,)
    assert out_shape2.to_numpy() == (2,)


def test_argmax_invalid_arg():
    a = np.arange(6).reshape(2, 3)
    op = ArgMax(axis=4)
    with pytest.raises(ValueError) as ex:
        op.check_incoming_shapes(StaticShape.from_tuple(a.shape))
    assert (
        str(ex.value) == "axis is out of bounds of shape "
        "(Dim.of(2), Dim.of(3)) got 4"
    )
