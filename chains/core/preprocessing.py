import numpy as np

__all__ = ["normalize"]


def normalize(x, axis=-1):
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / sigma


def one_hot(labels, cnt_classes, *, sample_axis=-1):
    return np.take(np.eye(cnt_classes), np.ravel(labels), axis=sample_axis)
