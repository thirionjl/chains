import math
from typing import Callable

import numpy as np

from ..core.graph import Graph
from ..core.optimizers import Optimizer
from ..core.tensor import Tensor
from ..utils import validate

__all__ = [
    "FeedMethod",
    "TrainListener",
    "Training",
    "MiniBatchTraining",
    "BatchTraining",
]

FeedMethod = Callable[[Tensor, Tensor], None]


class TrainListener:
    def on_start(self):
        pass

    def on_end(self):
        pass

    def on_epoch_start(self, epoch_num):
        pass

    def on_epoch_end(self, epoch_num, cost):
        pass

    def on_iteration(self, epoch_num, iteration, cost):
        pass


class Training:
    def __init__(self, optimizer: Optimizer, listener: TrainListener = TrainListener()):
        validate.is_a("listener", listener, TrainListener)
        validate.is_a("optimizer", optimizer, Optimizer)
        self.listener = listener
        self.optimizer = optimizer

    def train(
        self,
        cost_graph: Graph,
        feed_method: FeedMethod,
        x_train: Tensor,
        y_train: Tensor,
        epochs: int,
    ):
        validate.is_a("cost_graph", cost_graph, Graph)
        validate.is_callable("feed_method", feed_method)
        validate.is_tensor("x_train", x_train)
        validate.is_tensor("y_train", y_train)
        validate.is_strictly_greater_than("epochs", epochs, low=0)


class MiniBatchTraining(Training):
    batch_sizes = tuple(2**i for i in range(1, 11))

    def __init__(self, optimizer, listener, batch_size=64, sample_axis=-1):
        super().__init__(optimizer, listener)
        validate.is_one_of("batch_size", batch_size, self.batch_sizes)
        validate.is_one_of("sample_axis", sample_axis, (0, -1))

        self.batch_size = batch_size
        self.sample_axis = sample_axis

    def train(self, cost_graph, feed_method, x_train, y_train, epochs):
        super().train(cost_graph, feed_method, x_train, y_train, epochs)

        self.listener.on_start()
        self._train_init(cost_graph, feed_method, x_train, y_train)

        cnt_samples = x_train.shape[self.sample_axis]
        slices = self._batch_slices(
            cnt_samples, x_train.ndim, self.batch_size, self.sample_axis
        )

        for epoch in range(epochs):
            self.listener.on_epoch_start(epoch)
            epoch_cost = self._train_epoch(
                epoch, slices, feed_method, x_train, y_train, cnt_samples
            )
            self.listener.on_epoch_end(epoch, epoch_cost)

        self.listener.on_end()

    def _train_init(self, cost_graph, feed_method, x_train, y_train):
        cost_graph.initialize_variables()
        feed_method(x_train, y_train)
        self.optimizer.prepare_and_check(cost_graph)

    def _train_epoch(self, epoch, batch_slices, feed_method, x, y, cnt_samples):
        epoch_cost = 0
        cnt_batches = len(batch_slices)
        x_shuffled, y_shuffled = self._shuffle(x, y, cnt_samples)

        iteration = epoch * cnt_batches
        for batch_slice in batch_slices:
            feed_method(x_shuffled[batch_slice], y_shuffled[batch_slice])
            self.optimizer.run()
            epoch_cost += self.optimizer.cost / cnt_batches
            self.listener.on_iteration(epoch, iteration, self.optimizer.cost)
            iteration += 1
        return epoch_cost

    def _shuffle(self, x, y, cnt_examples):
        perm = np.random.permutation(cnt_examples)
        x_shuffled = self._shuffle_sample_axis(x, perm)
        y_shuffled = self._shuffle_sample_axis(y, perm)
        return x_shuffled, y_shuffled

    def _shuffle_sample_axis(self, x, perm):
        assert self.sample_axis in (-1, 0)

        if self.sample_axis == 0:
            return x[perm, ...]
        else:
            return x[..., perm]

    @staticmethod
    def _batch_slices(m, ndim, batch_size, axis):
        cnt_batches = math.ceil(m / batch_size)

        slices = []
        for i in range(cnt_batches):
            start = i * batch_size
            stop = min(m, (i + 1) * batch_size)
            slices.append(MiniBatchTraining._slice(axis, ndim, start, stop))

        return slices

    @staticmethod
    def _slice(axis, ndim, start, stop):
        assert axis in (-1, 0)

        axis_idx = axis if axis >= 0 else ndim + axis
        obj = []
        for dim in range(ndim):
            if dim == axis_idx:
                obj.append(slice(start, stop))
            else:
                obj.append(slice(None))
        return tuple(obj)


class BatchTraining(Training):
    def train(self, cost_graph, feed_method, x_train, y_train, epochs):
        super().train(cost_graph, feed_method, x_train, y_train, epochs)

        self.listener.on_start()
        self._train_init(cost_graph, feed_method, x_train, y_train)

        for i in range(epochs):
            self.listener.on_epoch_start(i)
            self.optimizer.run()
            cost = self.optimizer.cost
            self.listener.on_iteration(i, i, cost)
            self.listener.on_epoch_end(i, cost)

        self.listener.on_end()

    def _train_init(self, cost_graph, feed_method, x_train, y_train):
        cost_graph.initialize_variables()
        feed_method(x_train, y_train)
        self.optimizer.prepare_and_check(cost_graph)
