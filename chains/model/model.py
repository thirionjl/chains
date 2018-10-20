from chains.model.network import Network
from chains.model.training import Training


# TODO add configurable metrics: -> Accuracy ? Cost ?
# TODO Add accuracy method ?
class Model:

    def __init__(self, network: Network, training: Training):
        self.network = network
        self.training = training

    def train(self, x_train, y_train, *, epochs):
        self.training.train(self.network.cost_graph,
                            self.network.feed_cost_graph, x_train, y_train,
                            epochs)

    def predict(self, x_test):
        return self.network.evaluate(x_test)
