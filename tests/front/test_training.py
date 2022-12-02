from typing import Dict

import numpy as np

from chains.core import env
from chains.core import node_factory as nf
from chains.core.graph import Graph, Node
from chains.core.optimizers import Optimizer
from chains.utils.nd_typing import NdArrayLike
from chains.front.training import BatchTraining
from chains.front.training import MiniBatchTraining, TrainListener


class DummyListener(TrainListener):
    def __init__(self):
        self.calls = []

    def on_start(self):
        self.calls.append("on_start")

    def on_end(self):
        self.calls.append("on_end")

    def on_epoch_start(self, epoch_num):
        self.calls.append(f"on_epoch_start({epoch_num})")

    def on_epoch_end(self, epoch_num, cost):
        self.calls.append(f"on_epoch_end({epoch_num}, {cost})")

    def on_iteration(self, epoch_num, iteration, cost):
        self.calls.append(f"on_iteration({epoch_num}, {iteration}, {cost})")


class DummyOptimizer(Optimizer):
    def __init__(self, cost):
        self.cost = cost

    def prepare_and_check(self, graph: Graph):
        pass

    def run(self):
        self.cost -= 1

    def apply_gradients(self, gradient: Dict[Node, NdArrayLike]):
        pass


class DummyFeedMethod:
    def __init__(self):
        self.arg_pairs = []

    def __call__(self, x, y):
        self.arg_pairs.append((x, y))


def test_slice():
    s1 = MiniBatchTraining._slice(axis=0, ndim=4, start=1, stop=3)
    s2 = MiniBatchTraining._slice(axis=-1, ndim=4, start=1, stop=3)
    assert s1 == (slice(1, 3), slice(None), slice(None), slice(None))
    assert s2 == (slice(None), slice(None), slice(None), slice(1, 3))


def test_batch_slices():
    slices = MiniBatchTraining._batch_slices(m=5, ndim=3, batch_size=2, axis=-1)

    assert len(slices) == 3
    assert slices[0] == (slice(None), slice(None), slice(0, 2))
    assert slices[1] == (slice(None), slice(None), slice(2, 4))
    assert slices[2] == (slice(None), slice(None), slice(4, 5))

    slices = MiniBatchTraining._batch_slices(m=6, ndim=2, batch_size=3, axis=0)

    assert len(slices) == 2
    assert slices[0] == (slice(0, 3), slice(None))
    assert slices[1] == (slice(3, 6), slice(None))


def test_mini_batch_training():
    env.seed(1)
    g = Graph(nf.constant(8))
    x = np.arange(5)
    y = np.arange(5)
    feed = DummyFeedMethod()
    optimizer = DummyOptimizer(10)
    listener = DummyListener()
    train = MiniBatchTraining(optimizer, listener, batch_size=2)
    train.train(g, feed, x, y, 2)

    # Dry run
    assert_pair(feed.arg_pairs[0], x, y)
    # First epoch
    assert_pair(feed.arg_pairs[1], np.array([2, 1]), np.array([2, 1]))
    assert_pair(feed.arg_pairs[2], np.array([4, 0]), np.array([4, 0]))
    assert_pair(feed.arg_pairs[3], np.array([3]), np.array([3]))

    # Second epoch
    assert_pair(feed.arg_pairs[4], np.array([0, 2]), np.array([0, 2]))
    assert_pair(feed.arg_pairs[5], np.array([4, 3]), np.array([4, 3]))
    assert_pair(feed.arg_pairs[6], np.array([1]), np.array([1]))

    # 6 gradient descent steps
    assert optimizer.cost == 4

    assert len(listener.calls) == 12
    assert listener.calls[0] == "on_start"
    assert listener.calls[1] == "on_epoch_start(0)"
    assert listener.calls[2] == "on_iteration(0, 0, 9)"
    assert listener.calls[3] == "on_iteration(0, 1, 8)"
    assert listener.calls[4] == "on_iteration(0, 2, 7)"
    assert listener.calls[5] == "on_epoch_end(0, 8.0)"
    assert listener.calls[6] == "on_epoch_start(1)"
    assert listener.calls[7] == "on_iteration(1, 3, 6)"
    assert listener.calls[8] == "on_iteration(1, 4, 5)"
    assert listener.calls[9] == "on_iteration(1, 5, 4)"
    assert listener.calls[10] == "on_epoch_end(1, 5.0)"
    assert listener.calls[11] == "on_end"


def assert_pair(p, x, y):
    np.testing.assert_array_equal(p[0], x)
    np.testing.assert_array_equal(p[1], y)


def test_batch_training():
    g = Graph(nf.constant(8))
    x = np.arange(5)
    y = np.arange(5)
    feed = DummyFeedMethod()
    optimizer = DummyOptimizer(10)
    listener = DummyListener()

    train = BatchTraining(optimizer, listener)
    train.train(g, feed, x, y, 2)

    # Feed one
    assert_pair(feed.arg_pairs[0], x, y)

    # 2 gradient descent steps: 1 per epoch
    assert optimizer.cost == 8

    assert len(listener.calls) == 8
    assert listener.calls[0] == "on_start"
    assert listener.calls[1] == "on_epoch_start(0)"
    assert listener.calls[2] == "on_iteration(0, 0, 9)"
    assert listener.calls[3] == "on_epoch_end(0, 9)"
    assert listener.calls[4] == "on_epoch_start(1)"
    assert listener.calls[5] == "on_iteration(1, 1, 8)"
    assert listener.calls[6] == "on_epoch_end(1, 8)"
    assert listener.calls[7] == "on_end"
