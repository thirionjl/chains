import numpy as np


def forward(X, W, b):
    return W @ X + b.reshape(W.shape[0], 1)


def backward(dZ, X, W, b):
    db = np.sum(dZ, axis=1, keepdims=True)
    dW = dZ @ X.T
    dX = W.T @ dZ
    return dX, dW, db
