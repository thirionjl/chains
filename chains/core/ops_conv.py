from typing import Iterable, Union

import numpy as np

from chains.core.ops import Op
from chains.core.shape import StaticShape
from chains.core.tensor import Tensor
from chains.utils import validate

__all__ = ["Conv2D"]


class ConvFormat:
    Transposable = Union[Tensor, StaticShape]
    Perm = Iterable[int]

    def __init__(self, features_perm: Perm, filters_perm: Perm):
        validate.is_permutation(features_perm, 4)
        validate.is_permutation(filters_perm, 4)
        self.features_perm = features_perm
        self.filters_perm = filters_perm

    def nchw(self, a: Transposable) -> Transposable:
        return a.transpose(self.features_perm)

    def nchw_inv(self, a: Transposable) -> Transposable:
        return a.transpose(self._inverse_feature_perm())

    def dchw(self, f: Transposable) -> Transposable:
        return f.transpose(self.filters_perm)

    def dchw_inv(self, f: Transposable) -> Transposable:
        return f.transpose(self._inverse_filter_perm())

    def _inverse_filter_perm(self: Perm) -> Perm:
        return self.inverse_perm(self.filters_perm)

    def _inverse_feature_perm(self: Perm) -> Perm:
        return self.inverse_perm(self.features_perm)

    @staticmethod
    def inverse_perm(perm):
        return np.argsort(perm)

    @staticmethod
    def apply_perm(f, perm):
        return tuple(f[perm[i]] for i in range(len(perm)))


NCHW = ConvFormat((0, 1, 2, 3), (0, 1, 2, 3))
CHWN = ConvFormat((3, 0, 1, 2), (0, 1, 2, 3))


class Conv2D(Op):

    def __init__(self, feature_derivative=True, padding=0, stride=1,
                 conv_format=NCHW):
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
        self.zc = None

    def check_incoming_shapes(self, features: StaticShape,
                              filters: StaticShape, b: StaticShape):

        if features.ndim != 4:
            raise ValueError(f"features should be a 4 dimensional tensor "
                             f"but got {features.ndim} dimensions")
        if filters.ndim != 4:
            raise ValueError(f"filters should be a 4 dimensional tensor "
                             f"but got {filters.ndim} dimensions")

        d, c, fh, fw = self.conv_format.dchw(filters)
        m, s, nh, nw = self.conv_format.nchw(features)

        if c != s:
            raise ValueError(f"Number of channels should be the same "
                             f"in features({s}) and filters({c})")

        if not (b.is_column()):
            raise ValueError(f"Bias should be a 2-D column vector, got {b}")

        if d != b[0]:
            raise ValueError(f"Number of bias should match number of filters "
                             f"but got {b[0]} and {d}")

    def compute_out_shape(self, features: StaticShape, filters: StaticShape,
                          biases: StaticShape) -> StaticShape:

        d, c, fh, fw = self.conv_format.dchw(filters)
        m, s, nh, nw = self.conv_format.nchw(features)
        out_h, out_w = _clip_positions_count(nh.value, nw.value, fh.value,
                                             fw.value, self.padding,
                                             self.stride)

        tup = self.conv_format.nchw_inv(StaticShape(m, d, out_h, out_w))
        return StaticShape.from_tuple(tup)

    def compute(self, features: Tensor, filters: Tensor, bias: Tensor):
        self.features = self.conv_format.to_nchw(features)
        self.filters = self.conv_format.to_dchw(filters)
        self.bias = bias

        xc = im2col_activations(features, filters, self.padding, self.stride)
        fc = im2col_filters(features, filters)
        zc = fc @ xc + bias

        self.xc = xc
        self.fc = fc
        self.zc = zc
        out = reshape_out(zc, features, filters, self.padding, self.stride)
        self.output = self.conv_format.undo_to_nchw(out)

    def partials(self, d_output):
        m, c, nh, nw, d, fh, fw = _all_dimensions(self.features, self.filters)
        mm, dd, cnt_h, cnt_w = d_output.shape

        assert mm == m, f"Number of examples im activations({m}) is " \
            f"different from number of examples in output derivatives({mm})"
        assert dd == d, f"Number of filters in input({d}) is " \
            f"different from number of filters in output derivatives({dd})"

        dzc = d_output.transpose(0, 1, 2, 3).reshape(d, m * cnt_h * cnt_w)
        db = np.sum(d_output, axis=1, keepdims=True)
        dfc = d_output @ self.features.T
        dxc = self.weights.T @ d_output if self.feature_derivative else 0

        df = dfc.reshape(c, c, fh, fw)
        dx = col2im_activations(dxc, self.features, self.features,
                                self.padding, self.stride)

        return self.conv_format.undo_to_nchw(
            dx), self.conv_format.undo_to_dchw(df), db


def _all_dimensions(activations, filters):
    if activations.ndim != 4:
        raise ValueError(f"activations should be a 4 dimensional tensor "
                         f"but got {activations.ndim} dimensions")
    if filters.ndim != 4:
        raise ValueError(f"filters should be a 4 dimensional tensor "
                         f"but got {filters.ndim} dimensions")

    d, c, fh, fw = filters.shape
    m, s, nh, nw = activations.shape

    if c != s:
        raise ValueError(f"Number of channels should be the same "
                         f"in activations({s}) and filters({c})")

    return m, c, nh, nw, d, fh, fw


# From https://github.com/wiseodd/hipsternet/blob/master/hipsternet/im2col.py
def _im2col_indices(nh, nw, fh, fw, c, padding=0, stride=1):
    out_h, out_w = _clip_positions_count(nh, nw, fh, fw, padding, stride)

    i_rows = np.tile(np.repeat(np.arange(fh), fw), c)
    i_cols = stride * np.repeat(np.arange(out_h), out_w)
    i = i_rows.reshape(-1, 1) + i_cols.reshape(1, -1)

    j_rows = np.tile(np.arange(fw), fh * c)
    j_cols = stride * np.tile(np.arange(out_w), out_h)
    j = j_rows.reshape(-1, 1) + j_cols.reshape(1, -1)

    k = np.repeat(np.arange(c), fh * fw).reshape(-1, 1)
    return k.astype(int), i.astype(int), j.astype(int)


def im2col_activations(activations, filters, padding=0, stride=1):
    """
       Return the 4-D activations tensor as a 2-D tensor XC
       (c * fh * fw, m * nb_clip_positions), where:
       M[:, i] represents the "clip" where a filter should be applied but
       flattened.
       i = sample_index * n + clip_position where:
       - n is the number of clipping positions
       - clip_position is the position of top-left corner, when scanning
       activations first by rows, then by columns

       :return: activations as a 2-D tensor with dimensions
       (filter_flattened_size, m * clipping_positions_cnt)
       """
    m, c, nh, nw, d, fh, fw = _all_dimensions(activations, filters)
    activations_padded = pad(activations, padding)

    channels, rows, cols = _im2col_indices(nh, nw, fh, fw, c, padding, stride)

    # im2col_3d.shape = (m, filter_flattened_size, clipping_positions_cnt)
    im2col_3d = activations_padded[:, channels, rows, cols]
    im2col_2d = im2col_3d.transpose(1, 2, 0).reshape(c * fh * fw, -1)
    return im2col_2d


def im2col_filters(activations, filters):
    """Returns filters reshape as a matrix (d, c * fh * fw)"""
    _, c, _, _, d, fh, fw = _all_dimensions(activations, filters)
    return filters.reshape(d, c * fh * fw)


def pad(activations, padding=0):
    if padding == 0:
        return activations
    else:
        p = padding
        return np.pad(activations, ((0, 0), (0, 0), (p, p), (p, p)),
                      mode='constant')


def _clip_positions_count(nh, nw, fh, fw, padding=0, stride=1):
    """
    Counts the number of positions where the filters can be applied on
    activations.

    :return: A pair (number of horizontal positions, number of vertical
    positions)
    """
    if padding >= fh or padding >= fw:
        raise ValueError(f"Padding({padding}) should be strictly smaller than"
                         f" filter height({fh}) and width({fw})")

    width = nw + 2 * padding - fw
    height = nh + 2 * padding - fh

    if height % stride != 0:
        raise ValueError(f"Padded height ({height}) should be divisible by "
                         f"stride ({stride} but is not")
    if width % stride != 0:
        raise ValueError(f"Padded width ({width}) should be divisible by "
                         f"stride ({stride} but is not")

    return height // stride + 1, width // stride + 1


def col2im_activations(dXC, activations, filters, padding=0, stride=1):
    m, c, nh, nw, d, fh, fw = _all_dimensions(activations, filters)

    h_padded, w_padded = nh + 2 * padding, nw + 2 * padding
    activations_padded = np.zeros((m, c, h_padded, w_padded), dtype=dXC.dtype)

    k, i, j = _im2col_indices(nh, nw, fh, fw, c, padding, stride)
    cols_reshaped = dXC.reshape(c * fh * fw, -1, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(activations_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        out = activations_padded
    else:
        out = activations_padded[:, :, padding:-padding, padding:-padding]

    assert out.shape == activations.shape
    return out


def col2im_filters(dFC, activations, filters):
    """Returns filters reshape as a matrix (d, c * fh * fw)"""
    _, c, _, _, d, fh, fw = _all_dimensions(activations, filters)
    assert dFC.shape == (d, c * fh * fw)
    return filters.reshape(d, c, fh, fw)


def reshape_out(zc, activations, filters, padding=0, stride=1):
    """Reshapes ZC output to Z"""
    m, c, nh, nw, d, fh, fw = _all_dimensions(activations, filters)
    cnt_h, cnt_w = _clip_positions_count(activations, filters, padding, stride)
    assert zc.shape == (d, m * cnt_h * cnt_w)
    return zc.reshape((d, m, cnt_h, cnt_w)).transpose(1, 0, 2, 3)


def reshape_out_back(dZ, activations, filters, stride=1):
    """Reshapes dZC to dZ"""
    (m, c, nh, nw, d, fh, fw) = _all_dimensions(activations, filters)
    (cnt_h, cnt_w) = _clip_positions_count(activations, filters, stride, 0)
    assert dZ.shape == (m, d, cnt_h, cnt_w)
    return dZ.transpose(1, 0, 2, 3).reshape((d, m * cnt_h * cnt_w))
