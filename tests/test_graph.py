import inspect

from chains.core.graph import Node
from chains.core.ops import Constant, Op
from chains.core.ops_arithmetic import Add


def test_forward():
    n1 = Node(Constant(1.0))
    n2 = Node(Constant(2.0))
    n3 = Node(Constant(3.0))
    add = Add()
    s = Node(add, [n1, n2, n3])

    assert s is not None


def test_backward():
    assert False


def test_shape():
    assert False
