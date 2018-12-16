"""
Helper functions to measure progress and quality results of your learning
"""
import numpy as np

__all__ = ["accuracy"]


def accuracy(predicted_labels, actual_labels):
    """Proportion of correct predictions relative to the number of samples

    :param predicted_labels: Single dimensional array with predicted classes
    :param actual_labels: Single dimensional array with actual classes
    :return: An accuracy percentage between 0 and 100
    """
    are_equal = np.equal(actual_labels, predicted_labels)
    return np.mean(are_equal) * 100
