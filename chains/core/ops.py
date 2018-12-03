"""Set of base classes to extend to create your own operations that can fit
in a computation `<chains.core.graph.Node>`"""
import abc

import numpy as np

from chains.utils import validate
from .initializers import VarInitializer
from .shape import StaticShape
from .tensor import Tensor
from .utils_broadcasting import remove_broadcasting

__all__ = ["Op", "UnaryOp", "BinaryOp", "Var", "Placeholder",
           "Constant", "ElementWiseBinaryOp", "ElementWiseUnaryOp"]


class Op(abc.ABC):
    """Parent class of all operations that can be held by a `Node`object.
    Represents a simple operation and must provide a `compute`method that
    calculates the output value given some inputs and an `partials` method
    that computes the partial derivative of a cost function relative to all its
    inputs given the partial derivative of this same cost function relative to
    its output.

    Public attributes:
    - output: The computed output of the operation.

    Because `compute`is always called before `partial` it is recommended `Op`
    subclasses do cache the results of intermediate computations that are
    useful to compute the partial derivatives afterwards.
    """

    def __init__(self):
        self.output = None

    def compute(self):
        """Computes the output ot the operation given some input `Tensor`
        values. Must store the result in `self.output` field"""
        pass

    def partials(self, d_output):
        """Returns an ordered list representing partial derivatives relative to
         each input (in same order they were received in the `compute`method).

         Each partial derivative represents the derivative ot a cost function
         relative to it's input, given the partial derivative of that same
         cost function relative to this `Op`output. This partial derivative is
         given in parameter `d_output`
        """
        raise RuntimeError(f"{self} does not support derivation")

    def check_incoming_shapes(self, *static_shapes):
        """Verifies the shapes of inputs are valid.
        :param static_shapes: Ordered list of `<chains.core.shape.StaticShape>`
        """
        pass

    def compute_out_shape(self, *static_shapes) -> StaticShape:
        """Computes the output `<chains.core.shape.StaticShape>`

        :param static_shapes: Ordered list of `<chains.core.shape.StaticShape>`
        :return: Resulting `<chains.core.shape.StaticShape>` for those inputs
        """
        raise NotImplementedError

    def compute_out_dtype(self, *dtypes):
        """Computes the output dtype given input dtypes.

        :param dtypes: Ordered list of dtypes of the inputs
        :return: Resulting dtype given those inputs
        """
        return np.result_type(*dtypes)


class UnaryOp(Op, abc.ABC):
    """Base class for all `Op` having 1 input parameter"""

    def __init__(self):
        super().__init__()
        self.x = None

    def compute(self, x):
        self.x = x


class BinaryOp(Op, abc.ABC):
    """Base class for all `Op` having 2 input parameters"""

    def __init__(self):
        super().__init__()
        self.x = None
        self.y = None

    def compute(self, x, y):
        self.x = x
        self.y = y


class _NoOp(Op):
    """Base class for all `Op` having zero input parameters"""

    def __init__(self, shape: tuple, dtype=np.float32):
        super().__init__()
        validate.is_a("shape", shape, tuple)
        validate.is_number_dtype(dtype)

        self.shape = StaticShape.from_tuple(shape)
        self.dtype = np.dtype(dtype)

    def compute_out_shape(self) -> StaticShape:
        return self.shape

    def compute_out_dtype(self, *dtypes):
        return self.dtype

    def check(self):
        if self.output is None:
            raise ValueError("A settable cannot be set with None")
        value_shape = StaticShape.of_tensor(self.output)
        if not value_shape.is_assignable_to(self.shape):
            raise ValueError(
                f"{type(self)} accepts values compatible with "
                f"shape {self.shape}, but got {value_shape}")
        if np.result_type(self.output, self.dtype) != np.dtype(self.dtype):
            raise TypeError(
                f"{type(self)} is configured to accept only dtype {self.dtype}"
                f", but got {self.output.dtype} that is not castable to dtype")


class Var(_NoOp):
    """Represents a variable"""

    def __init__(self, initializer: VarInitializer, shape: tuple,
                 dtype=np.float32):
        """Creates a variable.
        :param initializer: VarInitializer that will initialize the variable
        :param shape: `StaticShape`of the variable
        :Param dtype: (optional) dtype of the variable. float32 by default
        """
        super().__init__(shape, dtype)
        validate.is_a("var_initializer", initializer, VarInitializer)

        if StaticShape.from_tuple(shape).is_unknown():
            raise ValueError(
                "Var should have only known dimensions in declared shape")

        self.initializer = initializer

    def initialize_if_needed(self):
        """Triggers variable initialization if not already done"""
        if self.output is None:
            self.output = self.initializer.initialize(self.shape.to_numpy(),
                                                      self.dtype)

        self.check()


class Placeholder(_NoOp):
    """Represents a placeholder. A placeholder is a constant that will be
    provided later, during the training phase of machine learning"""
    pass


class Constant(_NoOp):
    """Represent a simple constant"""

    def __init__(self, value: Tensor, dtype=np.float32):
        super().__init__(StaticShape.of_tensor(value), dtype)
        self.output = np.array(value).astype(dtype)
        self.check()


class ElementWiseBinaryOp(BinaryOp, abc.ABC):
    """Base class for `Op`s that take 2 inputs. If those inputs are Tensors,
    child classes run the same function for each component of that Tensor
    element-wise. This makes derivatives simpler to calculate. Moreover
    this base class deals with to inputs that are not the same shape but are
    broadcastable to the same shape.
    """

    def check_incoming_shapes(self, x: StaticShape, y: StaticShape):
        if not x.is_broadcast_compatible(y):
            raise ValueError(
                f"Shapes {x} and {y} cannot be broadcast together")

    def compute_out_shape(self, x: StaticShape, y: StaticShape) -> StaticShape:
        return x.broadcast(y)

    def partials(self, d_output):
        dx, dy = self.simple_derivatives()
        return (
            remove_broadcasting(self.x, d_output * dx),
            remove_broadcasting(self.y, d_output * dy)
        )

    def simple_derivatives(self):
        raise NotImplementedError


class ElementWiseUnaryOp(UnaryOp, abc.ABC):
    """Base class for `Op`s that take 1 input. If this input is a Tensor,
   child classes run the same function for each component of that Tensor
   element-wise. This makes derivatives simpler to calculate.
   """

    def check_incoming_shapes(self, x: StaticShape):
        pass

    def compute_out_shape(self, x_shape: StaticShape) -> StaticShape:
        return x_shape

    def partials(self, d_output):
        dx = self.simple_derivative()
        return d_output * dx,

    def simple_derivative(self):
        raise NotImplementedError
