import numpy as np

from chains.tensor.tensor import Tensor


def remove_broadcasting(in_tensor: Tensor, d_output: Tensor):
    out_shape = np.shape(d_output)
    in_shape = np.shape(in_tensor)

    if in_shape == out_shape:
        return d_output
    else:
        return _sum_broadcast_positions(d_output, in_shape, out_shape)


def _sum_broadcast_positions(t: Tensor, in_shape: tuple, out_shape):
    pad = max(len(out_shape) - len(in_shape), 0)
    padded_mat_dims = ((1,) * pad) + in_shape
    zipped_shapes = enumerate(zip(padded_mat_dims, out_shape))
    axis = tuple(i for i, (m_dim, t_dim) in zipped_shapes if m_dim == 1 and t_dim > 1)
    if len(axis) == 0:
        axis = None
    summed = np.sum(t, axis=axis, keepdims=True)
    return np.reshape(summed, in_shape)
