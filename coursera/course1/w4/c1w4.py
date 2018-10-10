import matplotlib.pyplot as plt
import numpy as np

from coursera.utils import binary_accuracy
from coursera.course1.w4.dnn_app_utils_v3 import load_data
from graph import node_factory as f, structure as g
from initialization import variable_initializers as init
from layers import fully_connected as fc
from optimizer import gradient_descent as gd
from tensor.tensor import Dim


class DeepNNModel:

    def __init__(self, features_count, hidden_layers_sizes=[]):
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

    @staticmethod
    def fully_connected_layer(features, cnt_features, cnt_neurons, l):
        # Enorme difference entre RandomNormalInitializer avec et sans [Sqrt] => apprentissage très lent !
        weights = f.var("W" + str(l + 1), init.HeInitializer(), shape=(cnt_neurons, cnt_features))
        biases = f.var("b" + str(l + 1), init.ZeroInitializer(), shape=(cnt_neurons, 1))
        return fc.fully_connected(features, weights, biases)

    def train(self, x_train, y_train, *, num_iterations=5_000, learning_rate=0.0075, print_cost=False):
        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(self.cost_graph, learning_rate=learning_rate)
        costs = []
        for i in range(num_iterations + 1):
            optimizer.run()

            if i % 100 == 0:
                costs.append(optimizer.cost)

            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {optimizer.cost}")

        return costs

    def predict(self, x_test):
        self.prediction_graph.placeholders = {self.X: x_test}
        return self.prediction_graph.evaluate()





def show_image(i, im_classes, x, y):
    plt.imshow(x[i])
    plt.show()
    print("y = " + str(y[0, i]) + ". It's a " + im_classes[y[0, i]].decode("utf-8") + " picture.")





if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (5.0, 4.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Data Loading
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Data  visualization
    show_image(10, classes, train_x_orig, train_y)

    # Data preparation
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))

    train_x_flatten = train_x_orig.reshape(m_train, -1).T
    test_x_flatten = test_x_orig.reshape(m_test, -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    # 2-layer data model with 7 hidden units
    n_x = 12288  # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    layer_configurations = [
        [7],
        [20, 7, 5]
    ]

    for hidden_layer_dims in layer_configurations:
        print(f"\n\n>>> Testing with hidden layers of dimensions {hidden_layer_dims}")
        model = DeepNNModel(n_x, hidden_layer_dims)
        costs = model.train(train_x, train_y, print_cost=True)
        plot_costs(costs)

        # Predict
        train_predictions = model.predict(train_x)
        train_accuracy = binary_accuracy(actual=train_predictions, expected=train_y)
        print(f"Train accuracy = {train_accuracy}%")

        test_predictions = model.predict(test_x)
        test_accuracy = binary_accuracy(actual=test_predictions, expected=test_y)
        print(f"Test accuracy = {test_accuracy}%")
