from collections.abc import Sequence
from typing import Iterable

import numpy as np

from ..core import tensor


def is_not_blank(name: str, value: str):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Parameter {name} cannot be blank")


def is_not_none(name: str, value: object):
    if value is None:
        raise ValueError(f"Parameter {name} should not be empty")


def is_one_of(name: str, value: object, legal_values: Iterable):
    if value not in legal_values:
        raise ValueError(
            f"Parameter {name} should be one of {legal_values}, " f"but got {value}"
        )


def is_not_one_of(name: str, value: object, forbidden_values: Iterable):
    if value in forbidden_values:
        raise ValueError(
            f"Parameter {name} should not be one of "
            f"{forbidden_values}, but got {value}"
        )


def is_a(name: str, value: object, t: type):
    if not isinstance(value, t):
        raise TypeError(
            f"Parameter {name} must be a {t} but "
            f"{type(value)} is not a subclass of it"
        )


def is_callable(name: str, value: object):
    if not callable(value):
        raise ValueError(f"Parameter {name} must be callable")


def is_tensor(name: str, value: object):
    if not tensor.is_tensor(name):
        raise ValueError(
            f"Parameter {name} must be a tensor, but got {type(value)}"
        )


def is_strictly_greater_than(name: str, value: int, low: int):
    if value <= low:
        raise ValueError(
            f"Parameter {name} must be strictly less than " f"{min}, but got {value}"
        )


def is_number_dtype(dtype, name="dtype"):
    _can_be_cast_to(dtype, np.float64, name)


def is_integer_dtype(dtype, name="dtype"):
    _can_be_cast_to(dtype, np.int64, name)


def _can_be_cast_to(dtype, super_type, name):
    if not np.can_cast(dtype, super_type):
        raise TypeError(f"Parameter {name} should be convertible " f"to {super_type}")


def is_permutation(perm: Sequence[int], length):
    identity = range(length)
    if set(perm) != set(identity):
        raise ValueError(f"{perm} is not a permutation tuple of {identity}")
