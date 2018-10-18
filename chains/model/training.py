import math

import numpy as np

from chains.model.network import Network
from chains.optimizer.gradient_descent import Optimizer


class TrainListener:
    def on_start(self):
        pass

    def on_end(self):
        pass

    def on_epoch_start(self, epoch_num):
        pass

    def on_iteration(self, epoch_num, mini_batch_num, cost):
        pass


class Training:

    def __init__(self, optimizer: Optimizer, listener: TrainListener = TrainListener()):
        self.listener = listener
        self.optimizer = optimizer

    def train(self, network: Network, x_train, y_train, epochs):
        pass


class MiniBatchTraining(Training):
    def __init__(self, optimizer: Optimizer, listener: TrainListener, batch_size=64,
                 sample_axis=-1):
        super().__init__(optimizer, listener)
        self.batch_size = batch_size
        self.sample_axis = sample_axis

    def train(self, network: Network, x_train, y_train, epochs):
        self.listener.on_start()
        network.initialize_variables()
        network.feed(x_train, y_train)
        self.optimizer.initialize_and_check(network.cost_graph)

        cnt_examples = x_train.shape[self.sample_axis]

        j = 0
        shuffled_range = np.arange(cnt_examples)
        for epoch in range(epochs):
            self.listener.on_epoch_start(epoch)
            np.random.shuffle(shuffled_range)
            batches = self.batches_from_to(cnt_examples, self.batch_size)

            for i, (start_idx, stop_idx) in enumerate(batches):
                indices = shuffled_range[start_idx:stop_idx]
                x = np.take(x_train, indices, axis=self.sample_axis)
                y = np.take(y_train, indices, axis=self.sample_axis)
                network.feed(x, y)
                self.optimizer.run()
                j += 1
                self.listener.on_iteration(epoch, i, j, self.optimizer.cost)

        self.listener.on_end()

    @staticmethod
    def batches_from_to(m, batch_size):
        batches = math.ceil(m / batch_size)
        return [(i * batch_size, min(m, (i + 1) * batch_size))
                for i in range(batches)]


class BatchTraining(Training):

    def train(self, network: Network, x_train, y_train, epochs):
        self.listener.on_start()
        network.initialize_variables()
        network.feed(x_train, y_train)
        self.optimizer.initialize_and_check(network.cost_graph)

        for i in range(epochs):
            self.listener.on_epoch_start(i)
            self.optimizer.run()
            cost = self.optimizer.cost
            self.listener.on_iteration(i, 1, i, cost)

        self.listener.on_end()
