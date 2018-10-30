import numpy as np

__all__ = ["accuracy"]


def accuracy(predicted_labels, actual_labels):
    are_equal = np.equal(actual_labels, predicted_labels)
    return np.mean(are_equal) * 100
