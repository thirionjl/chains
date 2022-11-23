"""Data structure elements of the computation graph"""
import warnings
from collections import deque, defaultdict
from operator import attrgetter
from typing import Dict, Set

import numpy as np

from .ops import Var, Constant, Placeholder, Op
from .ops_arithmetic import Negate, Pow, Add, ConstMul, Mul
from .ops_mat import Transpose, MatMul
from .tensor import is_tensor, Tensor
from ..utils.naming import NameGenerator

__all__ = ["Node", "Graph"]


class Node:
    """A Node in the computation graph.

    It is just a datastructure element that does not do computations by itself.
    Instead, it holds a reference to the actual computation which is of type
    `Op <chains.core.ops.Op>` and holds references to the incoming nodes.

    Most often you do not need to create a node directly but it is recommended
    to use one of the factory methods of `<chains.core.node_factory>` for
    instance:

    >>> from chains.core import node_factory as nf
    >>> Node logits = ...
    >>> Node softmax = nf.softmax(logits)

    Also `Node` implements the +,-,*,/,**(exponentiation),@ (matrix mult),
    infix operators, as well a .T (transpose) operators making it easy
    to create nodes without much code. For instance:

    >>> x = nf.initialized_var('x', np.array([[0.07], [0.4], [0.1]]))
    >>> A = nf.constant(np.array([[2, -1, -2], [-1, 1, 0], [-2, 0, 4]]))
    >>> cost = x.T() @ A @ x + 1

    Public attributes available after construction:
    - op: The underlying operation held by this node
    - incoming_nodes: The list of incoming nodes
    - name: A computed or user provided name
    - shape: The estimated `StaticShape` of the computation
    - dtype: The estimated data type `dtype` of the computation
    - out_nodes: The inferred outgoing nodes

    Public attributes available after computation:
    - value: The computed value. Should be a `<core.tensor.Tensor>` and is
    available only if method `compute`has been called before.
    """

    _all_node_names: Set[str] = set()
    _name_generator = NameGenerator()

    def __init__(self, op: Op, incoming_nodes=[], name: str = None):
        """Constructor
        :param op: The underlying computation
        :param incoming_nodes: The ordered list of incoming nodes. This
        list should correspond to the list of inputs for the `op` parameter.
        Order is important.
        :param name: (optional) A name given to this node. Useful for
        visualization tools or debugging. It is recommended the name is unique
        within your application.
        """
        incoming_shapes = tuple(n.shape for n in incoming_nodes)
        incoming_dtypes = tuple(n.dtype for n in incoming_nodes)
        op.check_incoming_shapes(*incoming_shapes)
        self.shape = op.compute_out_shape(*incoming_shapes)
        self.dtype = np.dtype(op.compute_out_dtype(*incoming_dtypes))
        self.op = op
        self._op_type = self._op_type(op)
        self._name = Node._name(op, name)
        self.incoming_nodes = tuple(incoming_nodes)
        self.out_nodes = set()
        for node in self.incoming_nodes:
            node.out_nodes.add(self)

    @staticmethod
    def _op_type(op: Op) -> str:
        return type(op).__name__

    @classmethod
    def _name(cls, op: Op, suggested_name: str = None) -> str:
        if suggested_name is None:
            return Node._name_generator.generate(cls._op_type(op))
        else:
            return cls._unique_name(suggested_name)

    @classmethod
    def _unique_name(cls, name: str) -> bool:
        if name in cls._all_node_names:
            warnings.warn(f"Node name {name} is already used")
        return name

    def __str__(self):
        _in = [n.name for n in self.incoming_nodes]
        return (
            f"<{self._op_type} '{self._name}' shape={self.shape}, "
            f"dtype={self.dtype}, in={_in}>"
        )

    def __repr__(self):
        _in = [n.name for n in self.incoming_nodes]
        return (
            f"Node({self._op!r}, {_in!r}, name={self._name!r}), "
            f"dtype={self.dtype!r}"
        )

    def __add__(self, other):
        if is_tensor(other):
            return Node(Add(), [self, Node(Constant(other))])
        elif isinstance(other, Node):
            return Node(Add(), [self, other])
        else:
            self._raise_unsupported_data_type_for_operation(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if is_tensor(other):
            return Node(Add(), [self, Node(Constant(-other))])
        elif isinstance(other, Node):
            return Node(Add(), [self, Node(Negate(), [other])])
        else:
            self._raise_unsupported_data_type_for_operation("subtraction", other)

    def __rsub__(self, other):
        if is_tensor(other):
            return Node(Add(), [Node(Constant(other)), Node(Negate(), [self])])
        elif isinstance(other, Node):
            return Node(Add(), [other, Node(Negate(), [self])])
        else:
            self._raise_unsupported_data_type_for_operation("subtraction", other)

    def __mul__(self, other):
        if is_tensor(other):
            return Node(ConstMul(other), [self])
        elif isinstance(other, Node):
            return Node(Mul(), [self, other])
        else:
            self._raise_unsupported_data_type_for_operation("multiplication", other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if isinstance(other, int):
            return Node(Pow(other), [self])
        else:
            self._raise_unsupported_data_type_for_operation("power", other)

    def __neg__(self):
        return Node(Negate(), [self])

    def __matmul__(self, other):
        if is_tensor(other):
            return Node(MatMul(other), [self])
        elif isinstance(other, Node):
            return Node(MatMul(), [self, other])
        else:
            self._raise_unsupported_data_type_for_operation(
                "matrix multiplication", other
            )

    def T(self):
        """Matrix transposition"""
        return Node(Transpose(), [self])

    @staticmethod
    def _raise_unsupported_data_type_for_operation(op_name, other):
        raise ValueError(f"Type {type(other)} is not supported type for {op_name}")

    def compute(self):
        """Computes the value of the node. A prerequisite is that input nodes
        have their value already computes (Compute has been called on them).
        """
        args = map(lambda n: n.value, self.incoming_nodes)
        self.op.compute(*args)

    def compute_partial_derivatives(self, out_derivative):
        """Given the partial derivative of a cost function relative to the
        output value of this node, computes all the partial derivatives of this
        same cost function relative to all the input nodes outputs.

        :param out_derivative: partial derivative of a cost function relative
        to the output of this node's computation

        :return: A list of `<chains.core.tensor.Tensor>` objects representing
        the derivatives relative to all incoming node's outputs. The list is
        in same order as the incoming nodes.
        """
        return self.op.partials(out_derivative)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Value for this node. `compute` has to be called before or a value
        has to been manually set"""
        return self.op.output

    @value.setter
    def value(self, value):
        """Sets the value of this node."""
        # On critical path: do not add to much checks here
        self.op.output = value

    def check_is_set(self):
        """Checks the value has been computed or set"""
        return self.op.check()

    def is_var(self):
        """Is this node holding a reference to a variable?"""
        return isinstance(self.op, Var)

    def is_constant(self):
        """Is this node holding a reference to a constant?"""
        return isinstance(self.op, Constant)

    def is_placeholder(self):
        """Is this node holding a reference to a placeholder?"""
        return isinstance(self.op, Placeholder)

    def is_assignable(self):
        """Can this node be directly assigned a value?"""
        return self.is_var() or self.is_placeholder()

    def is_computation(self):
        """Is this node holding a reference to a computation?"""
        return not (self.is_var() or self.is_placeholder() or self.is_constant())

    def initialize_if_needed(self):
        """Make sens only if it is a node holding a reference to a variable.
        If it is the case, the variable initializer is called to initialize
        the variable in case it has no value. If it is not the case a exception
        is risen.
        """
        if self.is_var():
            self.op.initialize_if_needed()
        else:
            raise ValueError(
                f"Node {self} is not a variable and could not be initialized"
            )


class Graph:
    """Represents a computation graph.

    A computation graph basically refers to a root node which is supposed to
    represent the cost function to optimize. `Node` names must be unique within
    a computation graph. Failure to do so will raise an Error.

    Public attributes:
    - all_nodes: A list of nodes of the computation graph
    - variables: The list of nodes holding a variable ot the computation graph
    - placeholders: Writable list of placeholders
    - shape: The output shape of the graph (shape of root node)

    Public methods:
    - initialize_variables: Triggers initialization of all variables
    - placeholders: Sets the list of all placeholders
    - check_initialized: Cehecks placeholders have been set and variables
      initialized
    - `evaluate` or `forward`: Does a forward pass over the computation graph
      and it returns its value
    - backward: Executes a back-propagation over the graph and returns the
      partial derivatives of the cost function(root) relative to all variables
    """

    def __init__(self, r):
        self._root = r
        self._forward_path = self._forward_prop_path(r)
        self.all_nodes = sorted(self._forward_path, key=attrgetter("name"))
        self._verify_unique_names(self.all_nodes)
        self.variables = {n for n in self._forward_path if n.is_var()}
        self._placeholders = {n for n in self._forward_path if n.is_placeholder()}
        self._backward_path = self._backward_prop_path(
            self.variables, self._forward_path
        )

    @staticmethod
    def _forward_prop_path(root):
        return Graph._topological_sort([root], lambda n: n.incoming_nodes)

    @staticmethod
    def _backward_prop_path(variables, fw_path):
        var_dependencies = Graph._topological_sort(variables, lambda n: n.out_nodes)
        return [n for n in var_dependencies if n in fw_path and n.is_computation()]

    @staticmethod
    def _verify_unique_names(nodes):
        used_names = set()
        for n in nodes:
            if n.name in used_names:
                raise ValueError(
                    f"Node name {n.name} is used multiple times in same graph"
                )
            used_names.add(n.name)

    @staticmethod
    def _topological_sort(roots, children_fct):
        to_explore = deque(roots)
        processed = set()
        res = []

        while len(to_explore) > 0:
            n = to_explore[-1]

            unprocessed_dependencies = set(children_fct(n)) - processed
            if len(unprocessed_dependencies) == 0:
                if n not in processed:
                    res.append(n)
                    processed.add(n)
                to_explore.pop()

            to_explore += unprocessed_dependencies

        return res

    def evaluate(self):
        """Evaluates the value of the computation graph. Same as `forward`"""
        return self.forward()

    def check_initialized(self):
        """Checks all variables and placeholders have been initialized"""
        self._check_placeholders()
        self._check_variables()

    def _check_placeholders(self):
        for p in self._placeholders:
            p.check_is_set()

    def _check_variables(self):
        for p in self.variables:
            p.check_is_set()

    def forward(self):
        """Evaluates the value of the computation graph. Same as `evaluate`"""
        for n in self._forward_path:
            n.compute()

        return n.value

    def backward(self):
        """Runs back-propagation over the graph.
        :return: A dictionary where keys are the variable `Node` and values
        are the matching partial derivatives of the cost function relative to
        that variable.
        """
        out_derivatives = defaultdict(int)
        out_derivatives[self._root] = 1

        for n in self._backward_path:
            partials = n.compute_partial_derivatives(out_derivatives[n])
            for n_in, d_in in zip(n.incoming_nodes, partials):
                out_derivatives[n_in] += d_in

        return out_derivatives

    @property
    def placeholders(self):
        """List of all placeholder nodes in the graph"""
        return self._placeholders

    @placeholders.setter
    def placeholders(self, values: Dict[Node, Tensor]):
        """Sets all the placeholder nodes in the graph"""
        for p in self._placeholders:
            p.value = values.get(p)
        # Performance: self._check_placeholders() On critical execution path

    def initialize_variables(self):
        """Triggers initialization of all the variables in the graph"""
        # Sort initializations so that it is deterministic in
        # case of random initializers
        for v in sorted(self.variables, key=lambda n: n.name):
            v.initialize_if_needed()

    @property
    def shape(self):
        """The output shape of the graph"""
        return self._root.shape
