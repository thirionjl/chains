import numpy as np


class GradientDescentOptimizer:

    def __init__(self, graph, learning_rate=None):
        graph.check_initialized()
        self._graph = graph
        self._cost = self._graph.forward()
        self._save()

        if learning_rate is None:
            self.learning_rate = self.find_acceptable_learning_rate()
            print(f"Trying new learning rate {self.learning_rate}")
        else:
            self.learning_rate = learning_rate

    def _save(self):
        self._variables_before = dict((vn, np.copy(vn.value)) for vn in self._graph.variables)
        self._cost_before = self._cost

    def _restore(self):
        for var_node, var_value in self._variables_before.items():
            var_node.value = np.copy(var_value)
        self._cost = self._cost_before

    def run(self):
        c = self._graph.forward()
        gradient = self._graph.backward()
        alpha = self.learning_rate

        # self._save()

        self._apply_gradient(gradient, alpha)


        # if c > self._cost + sensibility:
        #     self._restore()
        #     smaller_lrs = self._candidate_lrs(max_lr=self.learning_rate)
        #     self.learning_rate = self.find_acceptable_learning_rate(list(smaller_lrs))
        #     print(f"Trying new learning rate {self.learning_rate}")
        #     c = self._graph.forward()

        self._cost = c
        return gradient, c

    def _apply_gradient(self, gradient, lr):
        for var_node in self._graph.variables:
            var_node.value += - lr * gradient[var_node]

    def find_acceptable_learning_rate(self, learning_rates=None, sensibility=1e-6):
        if learning_rates is None:
            learning_rates = list(self._candidate_lrs())

        initial_cost = self._graph.forward()
        gradient = self._graph.backward()

        costs = {lr: self._estimate_cost(gradient, lr) for lr in learning_rates}
        best_lr = min(costs, key=costs.get)

        if costs[best_lr] < initial_cost + sensibility:
            return best_lr
        else:
            raise RuntimeError("Learning rates exhausted. Could not get down cost function")

    def _estimate_cost(self, gradient, lr):
        self._save()
        self._apply_gradient(gradient, lr)
        c = self._graph.forward()
        self._restore()
        return c

    @staticmethod
    def _candidate_lrs(max_lr=0.25, min_lr=1e-6):
        dividers = GradientDescentOptimizer._cycle((2.5, 2, 2))
        lr = max_lr
        while lr >= min_lr:
            yield lr
            lr /= next(dividers)

    @staticmethod
    def _cycle(step_sizes):
        idx = 0
        n = len(step_sizes)

        while True:
            if idx >= n:
                idx = 0
            yield step_sizes[idx]
            idx += 1

    @property
    def cost(self):
        return self._cost
