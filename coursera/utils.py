import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable


def binary_accuracy(actual: np.ndarray, expected: np.ndarray):
    """
    Computes the percentage of matching elements in both arrays

    :param actual: An array with only 1's and 0's
    :param expected: The reference array with only 1's and 0's
    :return: A number between 0 and 100 corresponding of the percentage of matching 1's or 0's
    """
    return 100 - np.mean(np.abs(actual - expected)) * 100


def plot_costs(costs: Iterable[float], *, unit: int, learning_rate: float):
    """
    Plots the a cost graph as a function of the iteration number

    :param costs: An array of cost values
    :param unit:  The number of iterations between 2 cost values
    :param learning_rate: The learning rate used. Shown in the title
    """
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel(f"Iterations (per {unit})")
    plt.title(f"Learning rate {learning_rate}")
    plt.show()
