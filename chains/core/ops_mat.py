import abc

import numpy as np

from .ops import UnaryOp, BinaryOp
from .shape import StaticShape
from .tensor import Tensor

__all__ = ["DimOp", "AsScalar", "Transpose", "MatMul", "Reduction",
           "SumComponents", "AvgComponents", "MaxComponent", "Reshape"]


class DimOp(UnaryOp):

    def __init__(self, axis: int = -1):
        if not isinstance(axis, int):
            raise ValueError(f"axis should be an integer got {type(axis)}")
        self.axis = axis

    def check_incoming_shapes(self, x: StaticShape) -> StaticShape:
        x.check_axis_index(self.axis)

    def compute_out_shape(self, x: StaticShape):
        return StaticShape.scalar()

    def compute(self, x):
        self.output = np.shape(x)[self.axis]

    def partials(self, d_output):
        return 0,

    def compute_out_dtype(self, *dtypes):
        return np.int


class AsScalar(UnaryOp):  # TODO Check root of graph is scalar

    def check_incoming_shapes(self, x: StaticShape):
        if len([d for d in x if d.is_unknown()]) > 0:
            raise ValueError(
                "AsScalar cannot accept tensors with unknown dimensions")
        sz = x.size()
        if sz != 1:
            raise ValueError(
                f"AsScalar could only be used if input tensor contains 1 "
                f"element only got {sz}")

    def compute_out_shape(self, x: StaticShape) -> StaticShape:
        return StaticShape.scalar()

    def compute(self, x: Tensor):
        self.output = np.asscalar(x)

    def partials(self, d_output):
        return np.atleast_1d(d_output),


class Transpose(UnaryOp):

    def __init__(self, axes=None):
        if axes is not None:
            if set(axes) != set(range(len(axes))):
                raise ValueError("Invalid permutation of axes submitted")
        self.axes = axes

    def check_incoming_shapes(self, x: StaticShape):
        if self.axes is None:
            if len(x) < 2:
                raise ValueError(
                    f"Transpose {self.axes} requires at least 2-D "
                    f"got {len(x)}")
        elif len(x) != len(self.axes):
            raise ValueError(f"Transpose {self.axes} requires an input tensor"
                             f" with {len(self.axes)} dimensions \
            but got {len(x)}")

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        if self.axes is None:
            args = (x_shape[1], x_shape[0]) + x_shape[2:]
        else:
            args = (x_shape[i] for i in self.axes)
        return StaticShape(*args)

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = np.transpose(x, self.axes)

    def partials(self, d_output):
        return np.transpose(d_output, self.axes),


class MatMul(BinaryOp):

    def check_incoming_shapes(self, x: StaticShape, y: StaticShape):
        if len(x) != 2:
            raise ValueError(f"Matrix multiplication requires 2-D tensors \
            but got a {len(x)}-D tensor as left operand")
        if len(y) != 2:
            raise ValueError(f"Matrix multiplication requires 2-D tensors \
            but got a {len(y)}-D tensor as right operand")

        if x[1] != y[0]:
            raise ValueError(f"Matrix multiplication requires that number of "
                             f"columns of left operand equals \
            number of rows of right operand, but got {x[1]} and {y[0]}")

    def compute_out_shape(self, x: StaticShape, y: StaticShape) -> StaticShape:
        return StaticShape(x[0], y[1])

    def compute(self, x: Tensor, y: Tensor):
        super().compute(x, y)
        self.output = np.dot(x, y)

    def partials(self, d_output):
        return np.dot(d_output, self.y.T), np.dot(self.x.T, d_output)


class Reduction(UnaryOp, abc.ABC):

    def check_incoming_shapes(self, x_shape: StaticShape):
        pass

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        return StaticShape.scalar()


class SumComponents(Reduction):  # todo: Manage axis

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = np.sum(x)

    def partials(self, d_output):
        return np.full(np.shape(self.x), d_output),


class AvgComponents(Reduction):  # todo: Manage axis

    def __init__(self):
        self.size = None

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = np.mean(x)
        self.size = np.size(x)

    def partials(self, d_output):
        return np.full(np.shape(self.x), d_output) / self.size,


class MaxComponent(Reduction):  # todo: Manage axis

    def __init__(self):
        super().__init__()
        self.maxIndex = None

    def compute(self, x: Tensor):
        super().compute(x)
        self.maxIndex = np.unravel_index(np.argmax(x), x.shape)
        self.output = x[self.maxIndex]

    def partials(self, d_output):
        d = np.zeros(np.shape(self.x))
        d[self.maxIndex] = d_output
        return d,


class ArgMax(UnaryOp):  # todo: Manage axis

    def __init__(self, axis=0):
        super().__init__()  # TODO: Check incoming shape against axis
        self.axis = axis

    def compute(self, x: Tensor):
        super().compute(x)
        self.output = np.argmax(x, axis=self.axis)

    def partials(self, d_output):
        return np.zeros(np.shape(self.x)),

    def check_incoming_shapes(self, x_shape: StaticShape):
        x_shape.check_axis_index(self.axis)

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        out_shape = []
        for axis, dim in enumerate(x_shape):
            if axis != self.axis:
                out_shape.append(dim)

        return StaticShape.from_tuple(out_shape)

    def compute_out_dtype(self, *dtypes):
        return np.int64


class Reshape(UnaryOp):

    def check_incoming_shapes(self, x_shape: StaticShape):
        print("Warning: Cannot check incoming shapes for reshape operations")
        pass

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        return StaticShape()

    def __init__(self, shape: StaticShape):
        super().__init__()
        self.initial_shape = None
        self.shape = shape  # TODO Refuse None in shape ! Could also add
        # reshape as binary op for dynamic shape

    def compute(self, x: Tensor):
        super().compute(x)
        self.initial_shape = np.shape(x)
        self.output = np.reshape(x, newshape=self.shape.to_numpy())

    def partials(self, d_output: Tensor):
        return (np.reshape(np.copy(d_output), newshape=self.initial_shape)),
