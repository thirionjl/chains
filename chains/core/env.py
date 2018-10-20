import numpy as np

__all__ = ["seed"]


def seed(value: int):
    np.random.seed(value)
