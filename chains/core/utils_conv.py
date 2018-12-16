from typing import Union, Iterable

import numpy as np

from .static_shape import StaticShape
from .tensor import Tensor, Shape
from ..utils import validate


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
TensorFlowNHWC = ConvFormat((0, 3, 1, 2), (3, 2, 0, 1))


def im2col(activations: Tensor, filters_shape: Shape, padding: int = 0,
           stride: int = 1):
    """
       Return the 4-D activations tensor as a 2-D tensor XC
       (c * fh * fw, m * nb_clip_positions), where:
       M[:, i] represents the "clip" where a filter should be applied but
       flattened.
       i = sample_index * n + clip_position where:
       - n is the number of clipping positions
       - clip_position is the position of top-left corner, when scanning
       activations first by rows, then by columns

       :param activations: 4-D Tensor in NCHW format
       :param filters_shape: Shape of filter in DCHW tuple format
       :param padding: Common padding applied to left,right,top and bottom
       :param stride: Stride
       :return: activations as a 2-D tensor with dimensions
       (filter_flattened_size, m * clipping_positions_cnt)
       """
    m, c, nh, nw, _, fh, fw = _all_dimensions(activations.shape, filters_shape)
    activations_padded = pad(activations, padding)

    channels, rows, cols = _im2col_indices(nh, nw, fh, fw, c, padding, stride)

    # im2col_3d.shape == (m, filter_flattened_size, clipping_positions_cnt)
    im2col_3d = activations_padded[:, channels, rows, cols]
    im2col_2d = im2col_3d.transpose(1, 0, 2).reshape(c * fh * fw, -1)
    return im2col_2d


def col2im(cols: Tensor, activations_shape: Shape, filters_shape: Shape,
           padding: int = 0, stride: int = 1):
    """
      Return a 4-D activations tensor from the im2col tensor.
      Note that when a cell from the activations has been used k times
      in im2col 2D matrix the k values are summed back. (consequence of
       multi-variable derivation chain rule applied)
    """
    m, c, nh, nw, _, fh, fw = _all_dimensions(activations_shape, filters_shape)

    h_padded, w_padded = nh + 2 * padding, nw + 2 * padding
    activations_padded = np.zeros((m, c, h_padded, w_padded), dtype=cols.dtype)

    k, i, j = _im2col_indices(nh, nw, fh, fw, c, padding, stride)
    cols_reshaped = cols.reshape(c * fh * fw, m, -1)
    cols_reshaped = cols_reshaped.transpose(1, 0, 2)
    np.add.at(activations_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        out = activations_padded
    else:
        out = activations_padded[:, :, padding:-padding, padding:-padding]

    assert out.shape == activations_shape
    return out


def pad(activations: Tensor, padding: int = 0):
    if padding == 0:
        return activations
    else:
        p = padding
        return np.pad(activations, ((0, 0), (0, 0), (p, p), (p, p)),
                      mode='constant')


def _all_dimensions(activations_shape: Shape, filters_shape: Shape):
    if len(activations_shape) != 4:
        raise ValueError(f"activations_shape should be 4 dimensional "
                         f"but got {len(activations_shape)} dimensions")
    if len(filters_shape) != 4:
        raise ValueError(f"filter_shape should be a dimensional "
                         f"but got {len(filters_shape)} dimensions")

    d, c, fh, fw = filters_shape
    m, s, nh, nw = activations_shape

    if c != s:
        raise ValueError(f"Number of channels should be the same "
                         f"in activations_shape({s}) and filters_shape({c})")

    return m, c, nh, nw, d, fh, fw


# Idea taken from:
# https://github.com/wiseodd/hipsternet/blob/master/hipsternet/im2col.py
def _im2col_indices(nh, nw, fh, fw, c, padding=0, stride=1):
    out_h, out_w = clip_positions_count(nh, nw, fh, fw, padding, stride)

    i_rows = np.tile(np.repeat(np.arange(fh), fw), c)
    i_cols = stride * np.repeat(np.arange(out_h), out_w)
    i = i_rows.reshape(-1, 1) + i_cols.reshape(1, -1)

    j_rows = np.tile(np.arange(fw), fh * c)
    j_cols = stride * np.tile(np.arange(out_w), out_h)
    j = j_rows.reshape(-1, 1) + j_cols.reshape(1, -1)

    k = np.repeat(np.arange(c), fh * fw).reshape(-1, 1)
    return k.astype(int), i.astype(int), j.astype(int)


def clip_positions_count(nh, nw, fh, fw, padding=0, stride=1):
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


def im2col_filters(filters: Tensor):
    """Returns 4-D filters reshape as a 2-D matrix (d, c * fh * fw)"""
    d, c, fh, fw = filters.shape
    return filters.reshape(d, c * fh * fw)


def col2im_filters(d_col_filters: Tensor, filters_shape: Shape):
    return d_col_filters.reshape(filters_shape)
