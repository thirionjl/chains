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

    def on_epoch(self, epoch_num):
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
        shuffled_range = np.arange(cnt_examples)
        for epoch in range(epochs):
            self.listener.on_epoch(epoch)
            np.random.shuffle(shuffled_range)
            batches = self.batches_from_to(cnt_examples, self.batch_size)

            for i, (start_idx, stop_idx) in enumerate(batches):
                indices = shuffled_range[start_idx:stop_idx]
                x = np.take(x_train, indices, axis=self.sample_axis)
                y = np.take(y_train, indices, axis=self.sample_axis)
                feed_method(x, y)
                self.optimizer.run()
                self.listener.on_iteration(epoch, i, j, self.optimizer.cost)
                j += 1

        self.listener.on_end()

    @staticmethod
    def batches_from_to(m, batch_size):
        batches = math.ceil(m / batch_size)
        return [(i * batch_size, min(m, (i + 1) * batch_size))
                for i in range(batches)]


class BatchTraining(Training):

    def train(self, cost_graph, feed_method, x_train, y_train, epochs):
        self.listener.on_start()
        cost_graph.initialize_variables()
        feed_method(x_train, y_train)
        self.optimizer.initialize_and_check(cost_graph)

        for i in range(epochs):
            self.listener.on_epoch(i)
            self.optimizer.run()
            cost = self.optimizer.cost
            self.listener.on_iteration(i, 0, i, cost)

        self.listener.on_end()
