import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[0, :].ravel(), X[1, :].ravel(), c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()


def load_2D_dataset():
    data = scipy.io.loadmat("datasets/data.mat")
    train_X = data["X"].T
    train_Y = data["y"].T
    test_X = data["Xval"].T
    test_Y = data["yval"].T

    plt.scatter(
        train_X[0, :].ravel(),
        train_X[1, :].ravel(),
        c=train_Y.ravel(),
        s=40,
        cmap=plt.cm.Spectral,
    )

    return train_X, train_Y, test_X, test_Y
