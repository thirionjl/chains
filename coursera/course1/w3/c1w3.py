import matplotlib.pyplot as plt
import numpy as np

from coursera.course1.w3.planar_utils import plot_decision_boundary, load_planar_dataset
from graph import node_factory as f, structure as g
from initialization import variable_initializers as init
from layers import fully_connected as fc
from optimizer import gradient_descent as gd
from tensor.tensor import Dim


class ShallowNNModel:

    def __init__(self, features_count, hidden_layers_size):
        # Number of examples
        self.m = Dim.unknown()
        # Number of features
        self.n = features_count
        # Number of hidden units
        self.h = hidden_layers_size

        # Placeholders and Vars
        self.W1 = f.var("W1", init.RandomNormalInitializer(), shape=(self.h, self.n))
        self.b1 = f.var("b1", init.ZeroInitializer(), shape=(self.h, 1))
        self.W2 = f.var("W2", init.RandomNormalInitializer(), shape=(1, self.h))
        self.b2 = f.var("b2", init.ZeroInitializer(), shape=(1, 1))
        self.X = f.placeholder(shape=(self.n, self.m))
        self.Y = f.placeholder(shape=(1, self.m))

        # Nodes
        lin_1 = fc.fully_connected(self.X, self.W1, self.b1)
        act_1 = f.tanh(lin_1)
        lin_2 = fc.fully_connected(act_1, self.W2, self.b2)
        loss = f.sigmoid_cross_entropy(lin_2, self.Y)
        predictions = f.is_greater_than(f.sigmoid(lin_2), 0.5)

        # Computation graphs
        self.cost_graph = g.Graph(loss)
        self.prediction_graph = g.Graph(predictions)

    def train(self, x_train, y_train, num_iterations=10000, learning_rate=20, print_cost=False):
        init.RandomNormalInitializer.seed(2)

        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(self.cost_graph, learning_rate=learning_rate)
        costs = []
        for i in range(num_iterations + 1):
            optimizer.run()

            if i % 100 == 0:
                costs.append(optimizer.cost)

            if i % 1_000 == 0:
                optimizer.learning_rate = optimizer.find_acceptable_learning_rate(
                    learning_rates=[20, 10, 5, 4, 2, 1, 0.5])

            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {optimizer.cost}")

        return costs

    def predict(self, x_test):
        self.prediction_graph.placeholders = {self.X: x_test}
        return self.prediction_graph.evaluate()


def accuracy(actual, expected):
    return 100 - np.mean(np.abs(actual - expected)) * 100


if __name__ == "__main__":
    # Dataset Loading
    X, Y = load_planar_dataset()
    shape_X = X.shape
    shape_Y = Y.shape
    m = shape_X[1]  # training set size

    # Test different hidden layer sizes
    plt.figure()
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50, 100]
    for i, h in enumerate(hidden_layer_sizes):
        print(f"\n\n>>> Testing with hidden layer of size {h}")

        # Model
        model = ShallowNNModel(X.shape[0], h)
        model.train(X, Y, num_iterations=5_000)

        # Predict
        train_predictions = model.predict(X)
        train_accuracy = accuracy(actual=train_predictions, expected=Y)
        print(f"Train accuracy = {train_accuracy}%")

        # Plot
        plt.subplot(5, 2, i + 1)
        plt.title(f"Hidden Layer of size {h}")
        plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
        plt.title(f"Decision Boundary for hidden layer size {h}")

    plt.show()
