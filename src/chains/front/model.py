from .network import Network
from .training import Training


# TODO add configurable metrics: -> Accuracy ? Cost ?
# TODO Add accuracy method ?
class Model:
    def __init__(self, network: Network, training: Training = None):
        self.network = network
        self.training = training

    def train(self, x_train, y_train, *, epochs, training=None):
        if training is not None:
            self.training = training
        if self.training is None:
            raise ValueError("Training method has not been set")

        self.training.train(
            self.network.cost_graph,
            self.network.feed_cost_graph,
            x_train,
            y_train,
            epochs,
        )

    def predict(self, x_test):
        return self.network.evaluate(x_test)
