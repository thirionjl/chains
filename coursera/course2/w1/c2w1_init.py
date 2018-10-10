import matplotlib.pyplot as plt
import numpy as np

from coursera.course2.w1.init_utils import load_dataset, plot_decision_boundary
from graph import node_factory as f, structure as g
from initialization import variable_initializers as init
from layers import fully_connected as fc
from optimizer import gradient_descent as gd
from tensor.tensor import Dim


class NNModel:
    inits = {
        "he": init.HeInitializer(seed=3),
        "random": init.RandomNormalInitializer(20, seed=3),
        "zeros": init.ZeroInitializer(),
    }

    def __init__(self, features_count, initializer_name, hidden_layers_sizes=[10, 5]):
        self.weight_initializer = self.inits.get(initializer_name)
        # Number of examples
        self.m = Dim.unknown()
        # Number of features
        self.n = features_count
        # Number of hidden units
        self.L = len(hidden_layers_sizes)

        # Placeholders and Vars
        self.X = f.placeholder(shape=(self.n, self.m))
        self.Y = f.placeholder(shape=(1, self.m))

        # Hidden layers
        a = self.X
        a_size = self.n
        for l, h in enumerate(hidden_layers_sizes):
            linear = self.fully_connected_layer(a, a_size, h, l)
            a = f.relu(linear)
            a_size = h

        # Output layer
        linear = self.fully_connected_layer(a, a_size, 1, self.L)
        loss = f.sigmoid_cross_entropy(linear, self.Y)
        predictions = f.is_greater_than(f.sigmoid(linear), 0.5)

        # Computation graphs
        self.cost_graph = g.Graph(loss)
        self.prediction_graph = g.Graph(predictions)

    def fully_connected_layer(self, features, cnt_features, cnt_neurons, l):
        weights = f.var("W" + str(l + 1), self.weight_initializer, shape=(cnt_neurons, cnt_features))
        biases = f.var("b" + str(l + 1), init.ZeroInitializer(), shape=(cnt_neurons, 1))
        return fc.fully_connected(features, weights, biases)

    def train(self, x_train, y_train, *,
              num_iterations=10_000,
              learning_rate=0.01,
              print_cost=True):

        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(self.cost_graph, learning_rate=learning_rate)
        costs = []
        for i in range(num_iterations + 1):
            optimizer.run()

            if i % 100 == 0:
                costs.append(optimizer.cost)

            if print_cost and i % 1000 == 0:
                print(f"Cost after iteration {i}: {optimizer.cost}")

        return costs

    def predict(self, x_test):
        self.prediction_graph.placeholders = {self.X: x_test}
        return self.prediction_graph.evaluate()


def accuracy(actual, expected):
    return 100 - np.mean(np.abs(actual - expected)) * 100


def show_image(i, im_classes, x, y):
    plt.imshow(x[i])
    plt.show()
    print("y = " + str(y[0, i]) + ". It's a " + im_classes[y[0, i]].decode("utf-8") + " picture.")


def plot_costs(costs):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 1000)')
    plt.title("Learning")
    plt.show()


def plot_boundary(init_name, m, xt, yt):
    plt.title(f"Model with {init_name} initializer")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: m.predict(x.T), xt, yt)


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load image dataset: blue/red dots in circles
    train_x, train_y, test_x, test_y = load_dataset()
    plt.show()

    m_train = train_x.shape[1]
    m_test = test_x.shape[1]
    n = train_x.shape[0]

    for initializer in NNModel.inits.keys():
        # Model
        model = NNModel(n, initializer)
        costs = model.train(train_x, train_y, print_cost=True)
        plot_costs(costs)

        # Predict
        train_predictions = model.predict(train_x)
        train_accuracy = accuracy(actual=train_predictions, expected=train_y)
        print(f"Train accuracy = {train_accuracy}%")

        test_predictions = model.predict(test_x)
        test_accuracy = accuracy(actual=test_predictions, expected=test_y)
        print(f"Test accuracy = {test_accuracy}%")

        # Plot
        plot_boundary(initializer, model, train_x, train_y)
