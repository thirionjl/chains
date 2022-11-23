"""Function to alter the execution environment"""
import numpy as np

__all__ = ["seed"]


def seed(value: int):
    """Seeds the random value generator"""
    np.random.seed(value)
