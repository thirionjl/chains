import math
from typing import Callable
import cProfile
import numpy as np

from chains.core.graph import Graph
from chains.core.optimizers import Optimizer
from chains.core.tensor import Tensor

__all__ = ["FeedMethod", "TrainListener", "Training", "MiniBatchTraining",
           "BatchTraining"]

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

    def on_iteration(self, epoch_num, mini_batch_num, iteration, cost):
        pass


class Training:

    def __init__(self, optimizer: Optimizer,
                 listener: TrainListener = TrainListener()):
        self.listener = listener
        self.optimizer = optimizer

    def train(self, cost_graph: Graph, feed_method: FeedMethod,
              x_train: Tensor, y_train: Tensor, epochs: int):
        pass


class MiniBatchTraining(Training):
    def __init__(self, optimizer, listener, batch_size=64, sample_axis=-1):
        super().__init__(optimizer, listener)
        self.batch_size = batch_size
        self.sample_axis = sample_axis

    def train(self, cost_graph, feed_method, x_train, y_train, epochs):
        self.listener.on_start()
        cost_graph.initialize_variables()
        feed_method(x_train, y_train)
        self.optimizer.initialize_and_check(cost_graph)

        cnt_samples = x_train.shape[self.sample_axis]
        batch_slices = self.compute_batch_slices(cnt_samples, x_train.ndim)

        pr = cProfile.Profile()


        for epoch in range(epochs):

            if epoch % 100 == 7:
                pr.clear()
                pr.enable()

            self.listener.on_epoch_start(epoch)
            epoch_cost = self.run_epoch(batch_slices, feed_method, x_train, y_train, cnt_samples)
            self.listener.on_epoch_end(epoch, epoch_cost)

            if epoch % 100 == 7:
                pr.dump_stats(f"training_{epoch}.prof")
                pr.create_stats()

        self.listener.on_end()

    def run_epoch(self, batch_slices, feed_method, x, y, cnt_samples):
        epoch_cost = 0
        cnt_batches = len(batch_slices)
        x_shuffled, y_shuffled = self.shuffle(x, y, cnt_samples)

        for batch_slice in batch_slices:
            feed_method(x_shuffled[batch_slice], y_shuffled[batch_slice])
            self.optimizer.run()
            epoch_cost += self.optimizer.cost / cnt_batches
        return epoch_cost

    def shuffle(self, x, y, cnt_examples):
        perm = np.random.permutation(cnt_examples)
        x_shuffled = x[self.slice_from_permutation(x.ndim, perm)]
        y_shuffled = y[self.slice_from_permutation(y.ndim, perm)]
        return x_shuffled, y_shuffled

    def compute_batch_slices(self, m, ndim):
        bs = self.batch_size
        batches = math.ceil(m / bs)
        bounds = [(i * bs, min(m, (i + 1) * bs)) for i in range(batches)]
        inds = [self.slice_from_boundaries(ndim, start, stop) for start, stop in bounds]
        return inds

    def slice_from_boundaries(self, ndim, start, stop):
        obj = []
        for dim in range(ndim):
            if dim == self.sample_axis or dim - self.sample_axis == ndim:
                obj.append(slice(start, stop))
            else:
                obj.append(slice(None))
        return tuple(obj)

    def slice_from_permutation(self, ndim, perm):
        obj = []
        for dim in range(ndim):
            if dim == self.sample_axis or dim - self.sample_axis == ndim:
                obj.append(perm)
            else:
                obj.append(slice(None))
        return tuple(obj)


class BatchTraining(Training):

    def train(self, cost_graph, feed_method, x_train, y_train, epochs):
        self.listener.on_start()
        cost_graph.initialize_variables()
        feed_method(x_train, y_train)
        self.optimizer.initialize_and_check(cost_graph)

        for i in range(epochs):
            self.listener.on_epoch_start(i)
            self.optimizer.run()
            cost = self.optimizer.cost
            self.listener.on_iteration(i, 0, i, cost)

        self.listener.on_end()
