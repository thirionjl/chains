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
    conv = Conv2D(padding=1, stride=2)
    m = Dim.unknown()

    features = StaticShape(m, 3, 40, 30)
    filters = StaticShape(16, 3, 9, 9)
    biases = StaticShape(16, 1)

    conv.check_incoming_shapes(features, filters, biases)
    out_shape = conv.compute_out_shape(features, filters, biases)
    assert out_shape == StaticShape(m, 16, 32, 22)
