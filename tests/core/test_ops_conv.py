import pytest

from chains.core.ops_conv import Conv2D
from chains.core.shape import StaticShape, Dim


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
