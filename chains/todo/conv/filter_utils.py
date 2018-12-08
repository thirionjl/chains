import abc

import numpy as np


class ConvFormat(abc.ABC):
    def all_dimensions(self, activations, filters):
        raise NotImplementedError


class NchwConvFormat(ConvFormat):
    def all_dimensions(self, activations, filters):
        """
        Returns a tuple with (cnt_samples, channels, height, width, number
        of filters, filter_height, filter_with)

        :param activations: Input activations matrix - 4 dimensional matrix
        :param filters: Filters - 4 dimensional matrix
        :return: Tuple with (number of examples, channels, height, width, number
        of filters, filter_height, filter_with)
        """
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


NCHW = NchwConvFormat()


def clip_positions_count(activations, filters, conv_format=NCHW, stride=1,
                         padding=0):
    """
    Counts the number of positions where the filters can be applied on
    activations.

    :return: A pair (number of horizontal positions, number of vertical
    positions)
    """
    _, _, nh, nw, _, fh, fw = conv_format.all_dimensions(activations, filters)
    if padding >= fh or padding >= fw:
        raise ValueError(f"Padding({padding}) should be strictly smaller than"
                         f" filter height({fh}) and width({fw})")

    return (nh + 2 * padding - fh) // stride + 1, (
        nw + 2 * padding - fw) // stride + 1


def im2col_activations(activations, filters, conv_format=NCHW, stride=1):
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
    m, c, nh, nw, d, fh, fw = conv_format.all_dimensions(activations, filters)
    flattened_filter_size = c * fh * fw
    cnt_h, cnt_w = clip_positions_count(activations, filters, stride, 0)
    columns = np.zeros((flattened_filter_size, m * cnt_h * cnt_w),
                       dtype=activations.dtype)

    idx = 0
    for sample_idx in range(0, m):
        for h in range(0, cnt_h):
            for w in range(0, cnt_w):
                h_start = h * stride
                w_start = w * stride
                clip = activations[
                       sample_idx,
                       :,
                       h_start:h_start + fh,
                       w_start:w_start + fw]
                columns[:, idx] = clip.reshape(-1)
                idx = idx + 1
    return columns


def im2col_activations_back(xc, activations, filters, conv_format=NCHW,
                            stride=1):
    m, c, nh, nw, d, fh, fw = conv_format.all_dimensions(activations, filters)
    cnt_h, cnt_w = clip_positions_count(activations, filters, stride, 0)
    flattened_filter_size = c * fh * fw
    assert xc.shape == (flattened_filter_size, m * cnt_h * cnt_w)

    x = np.zeros((m, c, nh, nw), dtype=activations.dtype)

    idx = 0
    cnt_clips = cnt_h * cnt_w
    for h_start in range(0, cnt_h):
        for w_start in range(0, cnt_w):
            h = h_start * stride
            w = w_start * stride
            clips = xc[:, idx::cnt_clips].reshape(c, fh, fw, m).transpose(3, 0,
                                                                          1, 2)
            x[:, :, h:h + fh, w:w + fw] += clips
            idx = idx + 1
    return x


def im2col_filters(activations, filters, conv_format=NCHW):
    """Returns filters reshape as a matrix (d, c * fh * fw)"""
    _, c, _, _, d, fh, fw = conv_format.all_dimensions(activations, filters)
    return filters.reshape(d, c * fh * fw)


def im2col_filters_back(dFC, activations, filters, conv_format=NCHW):
    """Returns filters reshape as a matrix (d, c * fh * fw)"""
    _, c, _, _, d, fh, fw = conv_format.all_dimensions(activations, filters)
    assert dFC.shape == (d, c * fh * fw)
    return filters.reshape(d, c, fh, fw)


def reshape_out(zc, activations, filters, conv_format=NCHW, stride=1):
    """Reshapes ZC output to Z"""
    m, c, nh, nw, d, fh, fw = conv_format.all_dimensions(activations, filters)
    cnt_h, cnt_w = clip_positions_count(activations, filters, stride, 0)
    assert zc.shape == (d, m * cnt_h * cnt_w)
    return zc.reshape((d, m, cnt_h, cnt_w)).transpose(1, 0, 2, 3)


def reshape_out_back(dZ, activations, filters, conv_format=NCHW, stride=1):
    """Reshapes dZC to dZ"""
    (m, c, nh, nw, d, fh, fw) = conv_format.all_dimensions(activations,
                                                           filters)
    (cnt_h, cnt_w) = clip_positions_count(activations, filters, stride, 0)
    assert dZ.shape == (m, d, cnt_h, cnt_w)
    return dZ.transpose(1, 0, 2, 3).reshape((d, m * cnt_h * cnt_w))
