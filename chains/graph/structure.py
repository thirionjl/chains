from collections import deque, defaultdict
from typing import Dict
from typing import Set

from chains.operations import arithmetic_ops as ops, mat_ops as mat, \
    base_ops as base
from chains.tensor import tensor as t


class NameGenerator:

    def __init__(self):
        self._max_id_per_type: Dict[str, int] = dict()

    def generate(self, category: str) -> str:
        seq = self._max_id_per_type.get(category, 0)
        self._max_id_per_type[category] = seq + 1
        return f"{category}:{seq}"


class Node:
    _name_generator = NameGenerator()
    _all_node_names: Set[str] = set()

    def __init__(self, op: base.Op, incoming_nodes=[], name: str = None):
        incoming_shapes = tuple(n.shape for n in incoming_nodes)
        op.check_incoming_shapes(*incoming_shapes)
        self.shape = op.compute_out_shape(*incoming_shapes)
        self._op = op
        self._op_type = self._op_type(op)
        self._name = Node._name(op, name)
        self.incoming_nodes = tuple(incoming_nodes)
        self._out_nodes = set()
        for node in self.incoming_nodes:
            node._out_nodes.add(self)

    @staticmethod
    def _op_type(op: base.Op) -> str:
        return type(op).__name__

    @classmethod
    def _name(cls, op: base.Op, suggested_name: str = None) -> str:
        if suggested_name is None:
            return Node._name_generator.generate(cls._op_type(op))
        else:
            return cls._unique_name(suggested_name)

    @classmethod
    def _unique_name(cls, name: str) -> bool:
        if name in cls._all_node_names:
            raise ValueError(f"Node name {name} is already used")
        return name

    def __str__(self):
        _in = [n.name for n in self.incoming_nodes]
        return f"<{self._op_type} '{self._name}' shape={self.shape}, in={_in}>"

    def __repr__(self):
        _in = [n.name for n in self.incoming_nodes]
        return f"Node({self._op!r}, {_in!r}, name={self._name!r})"

    def __add__(self, other):
        if t.is_tensor(other):
            return Node(ops.Add(), [self, Node(base.Constant(other))])
        elif isinstance(other, Node):
            return Node(ops.Add(), [self, other])
        else:
            self.raise_unsupported_data_type_for_operation(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if t.is_tensor(other):
            return Node(ops.Add(), [self, Node(base.Constant(-other))])
        elif isinstance(other, Node):
            return Node(ops.Add(), [self, Node(ops.Negate(), [other])])
        else:
            self.raise_unsupported_data_type_for_operation("subtraction",
                                                           other)

    def __rsub__(self, other):
        if t.is_tensor(other):
            return Node(ops.Add(), [Node(ops.Constant(other)),
                                    Node(ops.Negate(), [self])])
        elif isinstance(other, Node):
            return Node(ops.Add(), [other, Node(ops.Negate(), [self])])
        else:
            self.raise_unsupported_data_type_for_operation("subtraction",
                                                           other)

    def __mul__(self, other):
        if t.is_tensor(other):
            return Node(ops.ConstMul(other), [self])
        elif isinstance(other, Node):
            return Node(ops.Mul(), [self, other])
        else:
            self.raise_unsupported_data_type_for_operation("multiplication",
                                                           other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if isinstance(other, int):
            return Node(ops.Pow(other), [self])
        else:
            self.raise_unsupported_data_type_for_operation("power", other)

    def __neg__(self):
        return Node(ops.Negate(), [self])

    def __matmul__(self, other):
        if t.is_tensor(other):
            return Node(mat.MatMul(other), [self])
        elif isinstance(other, Node):
            return Node(mat.MatMul(), [self, other])
        else:
            self.raise_unsupported_data_type_for_operation(
                "matrix multiplication", other)

    def T(self):
        return Node(mat.Transpose(), [self])

    @staticmethod
    def raise_unsupported_data_type_for_operation(op_name, other):
        raise ValueError(
            f"Type {type(other)} is not supported type for {op_name}")

    def compute(self):
        args = map(lambda n: n.value, self.incoming_nodes)
        self._op.compute(*args)

    def compute_partial_derivatives(self, out_derivative):
        return self._op.partials(out_derivative)

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._op.output

    @value.setter
    def value(self, value):
        # On critical path: do not add to much checks here
        self._op.output = value

    def check_is_set(self):
        return self._op.check()

    @property
    def outgoing_nodes(self):
        return self._out_nodes

    def is_var(self):
        return isinstance(self._op, base.Var)

    def is_constant(self):
        return isinstance(self._op, base.Constant)

    def is_placeholder(self):
        return isinstance(self._op, base.Placeholder)

    def is_assignable(self):
        return self.is_var() or self.is_placeholder()

    def is_computation(self):
        return not (
            self.is_var() or self.is_placeholder() or self.is_constant())

    def initialize(self):
        if self.is_var():
            self._op.initialize()
        else:
            raise ValueError(
                f"Node {self} is not a variable and could not be initialized")


class Graph:

    def __init__(self, r):
        self._root = r
        self._forward_path = self._forward_prop_path(r)
        self.variables = {n for n in self._forward_path if n.is_var()}
        self._placeholders = {n for n in self._forward_path if
                              n.is_placeholder()}
        self._backward_path = self._backward_prop_path(self.variables,
                                                       self._forward_path)

    @staticmethod
    def _forward_prop_path(root):
        return Graph._topological_sort([root], lambda n: n.incoming_nodes)

    @staticmethod
    def _backward_prop_path(variables, fw_path):
        var_dependencies = Graph._topological_sort(variables,
                                                   lambda n: n.outgoing_nodes)
        return [n for n in var_dependencies if
                n in fw_path and n.is_computation()]

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
        return self.forward()

    def check_initialized(self):
        self._check_placeholders()
        self._check_variables()

    def _check_placeholders(self):
        for p in self._placeholders:
            p.check_is_set()

    def _check_variables(self):
        for p in self.variables:
            p.check_is_set()

    def forward(self):
        for n in self._forward_path:
            n.compute()

        return n.value

    def backward(self):
        out_derivatives = defaultdict(int)
        out_derivatives[self._root] = 1

        for n in self._backward_path:
            partials = n.compute_partial_derivatives(out_derivatives[n])
            for n_in, d_in in zip(n.incoming_nodes, partials):
                out_derivatives[n_in] += d_in

        return out_derivatives

    @property
    def placeholders(self):
        return self._placeholders

    @placeholders.setter
    def placeholders(self, values: Dict[Node, t.Tensor]):
        for p in self._placeholders:
            p.value = values.get(p)
        self._check_placeholders()

    def initialize_variables(self):
        # Sort initializations so that it is deterministic in
        # case of random initializers
        for v in sorted(self.variables, key=lambda n: n.name):
            v.initialize()
