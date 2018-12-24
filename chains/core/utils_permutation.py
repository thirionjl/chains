from typing import Iterable

import numpy as np

Perm = Iterable[int]


def inverse_perm(perm):
    return np.argsort(perm)
