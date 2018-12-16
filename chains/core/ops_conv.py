import numpy as np

from .ops import Op
from .static_shape import StaticShape
from .tensor import Tensor
from ..core import utils_conv as uc

__all__ = ["Conv2D"]


class Conv2D(Op):

    def __init__(self, feature_derivative=True, padding=0, stride=1,
                 conv_format=uc.NCHW):
        super().__init__()
        self.feature_derivative = feature_derivative
        self.conv_format = conv_format
        self.padding = padding
        self.stride = stride
        self.features = None
        self.filters = None
        self.bias = None
        self.xc = None
        self.fc = None

    def check_incoming_shapes(self, features: StaticShape,
                              filters: StaticShape, b: StaticShape):

        if features.ndim != 4:
            raise ValueError(f"features should be a 4 dimensional tensor "
                             f"but got {features.ndim} dimensions")
        if filters.ndim != 4:
            raise ValueError(f"filters should be a 4 dimensional tensor "
                             f"but got {filters.ndim} dimensions")

        d, c, fh, fw = self.conv_format.dchw(filters)
        m, cc, nh, nw = self.conv_format.nchw(features)
        p = self.padding
        s = self.stride

        if c != cc:
            raise ValueError(f"Number of channels should be the same "
                             f"in features({cc}) and filters({c})")

        if not (b.is_column()):
            raise ValueError(f"Bias should be a 2-D column vector, got {b}")

        if d != b[0]:
            raise ValueError(f"Number of bias should match number of filters "
                             f"but got {b[0]} and {d}")

        if (nh.value + 2 * p) % s != 0:
            raise ValueError(
                f"Padded height ({nh + 2 * p}) should be a multiple of "
                f"stride {cc} but is not")

        if (nw.value + 2 * p) % s != 0:
            raise ValueError(
                f"Padded width ({nh + 2 * p}) should be a multiple of "
                f"stride {cc} but is not")

    def compute_out_shape(self, features: StaticShape, filters: StaticShape,
                          biases: StaticShape) -> StaticShape:

        d, c, fh, fw = self.conv_format.dchw(filters)
        m, s, nh, nw = self.conv_format.nchw(features)
        out_h, out_w = uc.clip_positions_count(nh.value, nw.value, fh.value,
                                               fw.value, self.padding,
                                               self.stride)

        tup = self.conv_format.nchw_inv(StaticShape(m, d, out_h, out_w))
        return StaticShape.from_tuple(tup)

    def compute(self, features: Tensor, filters: Tensor, bias: Tensor):
        self.features = self.conv_format.nchw(features)
        self.filters = self.conv_format.dchw(filters)
        self.bias = bias

        self.xc = uc.im2col(self.features, self.filters.shape, self.padding,
                            self.stride)
        self.fc = uc.im2col_filters(self.filters)
        zc = self.fc @ self.xc + bias

        out = self._reshape_out(zc)
        self.output = self.conv_format.nchw_inv(out)

    def partials(self, d_output):
        m, c, nh, nw, d, fh, fw = self._dimensions()

        d_out = self.conv_format.nchw(d_output)
        mm, dd, cnt_h, cnt_w = d_out.shape

        assert mm == m, f"Number of examples im activations({m}) is " \
            f"different from number of examples in output derivatives({mm})"
        assert dd == d, f"Number of filters in input({d}) is " \
            f"different from number of filters in output derivatives({dd})"

        dzc = self._reshape_out_back(d_out)

        db = np.sum(dzc, axis=1, keepdims=True)
        dfc = dzc @ self.xc.T
        dxc = self.fc.T @ dzc if self.feature_derivative else 0

        df = uc.col2im_filters(dfc, self.filters.shape)
        dx = uc.col2im(dxc, self.features.shape, self.filters.shape,
                       self.padding, self.stride)

        return self.conv_format.nchw_inv(dx), self.conv_format.dchw_inv(df), db

    def _reshape_out(self, zc):
        """Reshapes ZC output to Z"""
        m, c, nh, nw, d, fh, fw = self._dimensions()
        cnt_h, cnt_w = uc.clip_positions_count(nh, nw, fh, fw, self.padding,
                                               self.stride)

        assert zc.shape == (d, m * cnt_h * cnt_w)
        return zc.reshape((d, m, cnt_h, cnt_w)).transpose(1, 0, 2, 3)

    def _reshape_out_back(self, dz):
        """Reshapes dZC to dZ"""
        m, c, nh, nw, d, fh, fw = self._dimensions()
        cnt_h, cnt_w = uc.clip_positions_count(nh, nw, fh, fw, self.padding,
                                               self.stride)

        assert dz.shape == (m, d, cnt_h, cnt_w)
        return dz.transpose(1, 0, 2, 3).reshape((d, m * cnt_h * cnt_w))

    def _dimensions(self):
        assert self.features.ndim == 4, f"activations should be a 4 " \
            f"dimensional tensor but got {self.v.ndim} dimensions"
        assert self.filters.ndim == 4, f"filters should be a 4 dimensional " \
            f"tensor but got {self.filters.ndim} dimensions"

        d, c, fh, fw = self.filters.shape
        m, s, nh, nw = self.features.shape

        assert c == s, f"Number of channels should be the same " \
            f"in activations({s}) and filters({c})"

        return m, c, nh, nw, d, fh, fw
