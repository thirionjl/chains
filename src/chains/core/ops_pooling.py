import numpy as np

from chains.utils import validate
from .ops import Op
from .shape import Shape
from chains.utils.nd_typing import NdArrayLike
from ..core import utils_conv as uc

__all__ = ["MaxPool"]


class MaxPool(Op):
    def __init__(self, stride=2, conv_format=uc.NCHW):
        super().__init__()
        validate.is_strictly_greater_than("stride", stride, 1)
        self.conv_format = conv_format
        self.stride = stride
        self.features_nchw = None
        self.max_indices = None
        self.filters_shape = None
        self.x_shape = None
        self.xc_shape = None

    def check_incoming_shapes(self, features: Shape):

        if features.ndim != 4:
            raise ValueError(
                f"features should be a 4 dimensional tensor "
                f"but got {features.ndim} dimensions"
            )

        m, c, nh, nw = self.conv_format.nchw(features)
        s = self.stride

        if nh.value % s != 0:
            raise ValueError(
                f"Height ({nh.value}) should be a multiple of stride {s} " f"but is not"
            )

        if nw.value % s != 0:
            raise ValueError(
                f"Width ({nw.value}) should be a multiple of stride {s} " f"but is not"
            )

    def compute_out_shape(self, features: Shape) -> Shape:

        m, c, nh, nw = self.conv_format.nchw(features)
        out_h, out_w = uc.clip_positions_count(
            nh.value, nw.value, self.stride, self.stride, 0, self.stride
        )

        tup = self.conv_format.nchw_inv(Shape.of(m, c, out_h, out_w))
        return Shape.from_tuple(tup)

    def compute(self, features: NdArrayLike):
        self.features_nchw = self.conv_format.nchw(features)
        m, c, nh, nw = self.features_nchw.shape
        out_h, out_w = uc.clip_positions_count(
            nh, nw, self.stride, self.stride, 0, self.stride
        )

        self.x_shape = (m * c, 1, nh, nw)
        self.filters_shape = (m, 1, self.stride, self.stride)

        x = np.reshape(self.features_nchw, self.x_shape)
        xc = uc.im2col(x, self.filters_shape, padding=0, stride=self.stride)
        self.xc_shape = xc.shape

        self.max_indices = np.argmax(xc, axis=0)
        out = xc[self.max_indices, range(self.max_indices.size)]
        out = out.reshape(m, c, out_h, out_w)

        self.output = self.conv_format.nchw_inv(out)

    def partials(self, d_output: NdArrayLike):
        m, c, nh, nw = self.features_nchw.shape
        d_out = self.conv_format.nchw(d_output)

        mm, dd, out_h, out_w = d_out.shape

        assert mm == m, (
            f"Number of examples im activations({m}) is "
            f"different from number of examples in output derivatives({mm})"
        )

        d_out_flat = np.ravel(d_out)

        d_xc = np.zeros(self.xc_shape)
        d_xc[self.max_indices, range(self.max_indices.size)] = d_out_flat

        dx = uc.col2im(
            d_xc, self.x_shape, self.filters_shape, padding=0, stride=self.stride
        )

        d_features = np.reshape(dx, self.features_nchw.shape)
        return (self.conv_format.nchw_inv(d_features),)
