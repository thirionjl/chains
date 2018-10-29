from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_costs(costs: Iterable[float], *, unit: int, learning_rate: float):
    """
    Plots the a cost graph as a function of the iteration number

    :param costs: An array of cost values
    :param unit:  The number of iterations between 2 cost values
    :param learning_rate: The learning rate used. Shown in the title
    """
    plt.clf()
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel(f"Iterations (per {unit})")
    plt.title(f"Learning rate {learning_rate}")
    plt.show()
