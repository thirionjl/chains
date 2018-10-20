import numpy as np

__all__ = ["normalize"]


def normalize(x, axis=-1):
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / sigma
