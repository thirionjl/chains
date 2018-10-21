import math
from typing import Callable

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

        cnt_examples = x_train.shape[self.sample_axis]

        j = 0
        perm = np.arange(cnt_examples)
        batches = self.batches_from_to(cnt_examples, x_train.ndim)
        cnt_batches = len(batches)
        for epoch in range(epochs):
            self.listener.on_epoch_start(epoch)
            epoch_cost = 0
            np.random.shuffle(perm)
            x_shuffled = x_train[self.to_index_perm(x_train.ndim, perm)]
            y_shuffled = y_train[self.to_index_perm(y_train.ndim, perm)]

            for i, indexes in enumerate(batches):
                x = x_shuffled[indexes]
                y = y_shuffled[indexes]
                feed_method(x, y)
                self.optimizer.run()
                self.listener.on_iteration(epoch, i, j, self.optimizer.cost)
                j += 1
                epoch_cost += self.optimizer.cost / cnt_batches

            self.listener.on_epoch_end(epoch, epoch_cost)

        self.listener.on_end()

    def to_index_perm(self, ndim, perm):
        obj = []
        for dim in range(ndim):
            if dim == self.sample_axis or dim - self.sample_axis == ndim:
                obj.append(perm)
            else:
                obj.append(slice(None))
        return tuple(obj)

    def batches_from_to(self, m, ndim):
        bs = self.batch_size
        batches = math.ceil(m / bs)
        bounds = [(i * bs, min(m, (i + 1) * bs)) for i in range(batches)]
        inds = [self.to_index_obj(ndim, start, stop) for start, stop in bounds]
        return inds

    def to_index_obj(self, ndim, start, stop):
        obj = []
        for dim in range(ndim):
            if dim == self.sample_axis or dim - self.sample_axis == ndim:
                obj.append(slice(start, stop))
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
