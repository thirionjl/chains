import numpy as np


def create_batches(features: np.array, labels: np.array, batch_size: int,
                   examples_dim=-1):
    if len(features) != len(labels):
        raise ValueError(f"Features input and labels ")
    pass
