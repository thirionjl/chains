from itertools import zip_longest
from typing import Union, Iterable

import numpy as np

from .tensor import Shape, Tensor
from ..utils import validate

__all__ = ["Dim", "StaticShape"]


class Dim:

    @staticmethod
    def unknown():
        return Dim(None)

    @staticmethod
    def of(value):
        return Dim(value)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if self is other:
            return True
        elif self.value is None or other.value is None:
            return False
        else:
            return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def is_unknown(self):
        return self.value is None

    def is_concrete(self):
        return self.value is not None

    def is_broadcast_compatible(self, other):
        return self.value == 1 \
               or other.value == 1 \
               or (self.is_concrete() and other.is_concrete() and
                   self.value == other.value)

    def is_assignable_to(self, other):
        return (self == other) or (self.is_concrete() and other.is_unknown())

    def broadcast(self, other):
        if self.is_unknown():
            return self
        elif other.is_unknown():
            return other
        elif self.value > other.value:
            return self
        else:
            return other

    def __str__(self):
        if self.is_unknown():
            return "?"
        else:
            return str(self.value)

    def __repr__(self):
        if self.is_unknown():
            return "Dim.unknown()"
        else:
            return f"Dim.of({self.value})"

    def __add__(self, other):
        if self.value is None or other.value:
            raise ValueError("Cannot add unknown dimension")
        return self.value + other.value


class StaticShape(tuple):

    def __new__(cls, *args):
        dim_args = tuple(arg if isinstance(arg, Dim)
                         else Dim(arg) for arg in args)
        return tuple.__new__(cls, dim_args)

    @classmethod
    def scalar(cls):
        return cls()

    @classmethod
    def from_tuple(cls, t: Shape):
        return cls(*t)

    @classmethod
    def of_tensor(cls, t: Tensor):
        return cls.from_tuple(np.shape(t))

    def is_broadcast_compatible(self, other):
        return all(m.is_broadcast_compatible(n) for m, n in
                   self.zip_dimensions(self, other))

    def reduce_along_axis(self, axis: Union[Iterable[int], int],
                          keep_dims=False):
        ax_tuple = (axis,) if isinstance(axis, int) else axis

        squeezed = [True] * len(self)
        for ax in ax_tuple:
            self.check_axis_index(ax)
            squeezed[ax] = False

        if keep_dims:
            return StaticShape.from_tuple(
                (d if keep else 1) for d, keep in zip(self, squeezed))
        else:
            return StaticShape.from_tuple(
                d for d, keep in zip(self, squeezed) if keep)

    @staticmethod
    def zip_dimensions(a, b):
        reversed_zip = list(zip_longest(reversed(a), reversed(b),
                                        fillvalue=Dim.of(1)))
        return reversed(reversed_zip)

    def broadcast(self, other):
        args = (m.broadcast(n) for m, n in self.zip_dimensions(self, other))
        return StaticShape(*args)

    def size(self):
        sz = 1
        for d in self:
            sz *= d.value
        return sz

    def to_numpy(self):
        return tuple(d.value for d in self)

    def is_assignable_to(self, other):
        if len(self) != len(other):
            return False

        return all(s.is_assignable_to(o) for s, o in zip(self, other))

    def __str__(self):
        if len(self) == 0:
            return 'scalar'
        else:
            return super().__str__()

    def is_concrete(self):
        return all(d.is_concrete() for d in self)

    def is_unknown(self):
        return not self.is_concrete()

    def is_scalar(self):
        return len(self) == 0

    def is_column(self):
        return self.ndim == 2 and self[1] == Dim.of(1)

    @property
    def ndim(self):
        return len(self)

    def check_axis_index(self, axis: int) -> bool:
        if axis >= 0 and not (0 <= axis < self.ndim):
            raise ValueError(f"axis is out of bounds of shape {self} "
                             f"got {axis}")
        if axis < 0 and not (-self.ndim <= axis < 0):
            raise ValueError(f"axis is out of bounds of shape {self} "
                             f"got {axis}")

    def transpose(self, *args):
        perm = args[0] if len(args) == 1 else tuple(args)
        validate.is_permutation(perm, len(self))
        return StaticShape(*(self[perm[i]] for i in range(len(perm))))
