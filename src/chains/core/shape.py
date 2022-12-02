from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Union, Optional, Iterator, overload

import numpy as np

from chains.utils.nd_typing import NdArrayLike, NdShape
from ..utils import validate

__all__ = ["Dim", "Shape"]


@dataclass(frozen=True, slots=True)
class Dim:
    value: Optional[int]

    @staticmethod
    def unknown():
        return Dim(None)

    @staticmethod
    def of(value: int):
        assert value > 0, "Dimension should be a positive integer"
        return Dim(value)

    def is_unknown(self):
        return self.value is None

    def is_concrete(self):
        return self.value is not None

    def is_broadcast_compatible(self, other: "Dim") -> bool:
        return self.value == 1 or other.value == 1 or self == other

    def is_assignable_to(self, other: "Dim") -> bool:
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

    def __eq__(self, other):
        if self is other:
            return True
        elif self.value is None or other.value is None:
            return False
        else:
            return self.value == other.value

    def __hash__(self):
        return hash(self.value)

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


@dataclass(frozen=True, eq=True, slots=True)
class Shape:
    dims: tuple[Dim, ...]

    @staticmethod
    def of(*args: Dim | int | None):
        return Shape(tuple(arg if isinstance(arg, Dim) else Dim(arg) for arg in args))

    @staticmethod
    def scalar():
        return Shape.of()

    @staticmethod
    def from_tuple(t: NdShape):
        return Shape.of(*t)

    @staticmethod
    def _from_it(t: Iterator[Dim]):
        return Shape.of(*t)

    @staticmethod
    def of_array_like(t: NdArrayLike):
        return Shape.from_tuple(np.shape(t))

    def is_broadcast_compatible(self, other: "Shape") -> bool:
        return all(m.is_broadcast_compatible(n) for m, n in self._zip_dims(self, other))

    def reduce_along_axis(self, axis: Union[Sequence[int], int], keep_dims=False):
        ax_tuple: Sequence[int] = (axis,) if isinstance(axis, int) else axis

        squeezed = [True] * len(self.dims)
        for ax in ax_tuple:
            self.check_axis_index(ax)
            squeezed[ax] = False

        if keep_dims:
            return Shape._from_it(
                (d if keep else Dim.of(1)) for d, keep in zip(self.dims, squeezed)
            )
        else:
            return Shape._from_it(d for d, keep in zip(self.dims, squeezed) if keep)

    @staticmethod
    def _zip_dims(a: "Shape", b: "Shape") -> Iterator[tuple[Dim, Dim]]:
        reversed_zip = list(
            zip_longest(reversed(a.dims), reversed(b.dims), fillvalue=Dim.of(1))
        )
        return reversed(reversed_zip)

    def broadcast(self: "Shape", other: "Shape") -> "Shape":
        args = (m.broadcast(n) for m, n in self._zip_dims(self, other))
        return Shape._from_it(args)

    def size(self) -> int:
        sz = 1
        for d in self.dims:
            if d.value is None:
                raise ValueError(
                    f"{self} has no size because not all dimensions are known"
                )
            sz *= d.value
        return sz

    def to_numpy(self) -> Sequence[int]:
        if self.has_unknown_dim():
            raise ValueError(
                f"{self!r} cannot be converted to tuple[int] not all dimensions are known"
            )
        return tuple(d.value for d in self)  # type: ignore

    def is_assignable_to(self, other: "Shape") -> bool:
        if len(self.dims) != len(other.dims):
            return False

        return all(s.is_assignable_to(o) for s, o in zip(self.dims, other.dims))

    def __len__(self):
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)

    @overload
    def __getitem__(self, item: int) -> Dim:
        ...

    @overload
    def __getitem__(self, item: slice) -> tuple[Dim, ...]:
        ...

    def __getitem__(self, item: int | slice) -> Dim | tuple[Dim, ...]:
        return self.dims.__getitem__(item)  # type: ignore

    def has_unknown_dim(self) -> bool:
        return any(d.is_unknown() for d in self.dims)

    def is_scalar(self) -> bool:
        return len(self.dims) == 0

    def is_column(self) -> bool:
        return len(self.dims) == 2 and self.dims[1] == Dim.of(1)

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def __str__(self):
        return "(" + ", ".join(str(d) for d in self.dims) + ")"

    def check_axis_index(self, axis: int) -> None:
        if axis >= 0 and not (0 <= axis < self.ndim):
            raise ValueError(f"axis is out of bounds of shape {str(self)} got {axis}")
        if axis < 0 and not (-self.ndim <= axis < 0):
            raise ValueError(f"axis is out of bounds of shape {str(self)} got {axis}")

    def transpose(self, *args: int | tuple[int]) -> "Shape":
        perm: tuple[int] = args[0] if len(args) == 1 else tuple(args)  # type: ignore
        validate.is_permutation(perm, len(self.dims))
        return Shape._from_it(self.dims[perm[i]] for i in range(len(perm)))
