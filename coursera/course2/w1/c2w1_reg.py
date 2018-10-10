# import packages
import cProfile
import time

from chains.graph import node_factory as f, structure as g
from chains.graph import node_factory as nf
from chains.initialization import variable_initializers as init
from chains.layers import fully_connected as fc
from chains.operations import regularization_ops as reg
from chains.optimizer import gradient_descent as gd
from chains.tensor.tensor import Dim
from coursera.course2.w1.reg_utils import *


class CourseraModel:

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.X = None
        self.Y = None
        self.layers_dims = [n, 20, 3, 1]
        self.parameters = None

    def prepare(self, X, Y):
        self.X = X
        self.Y = Y
        self.parameters = initialize_parameters(self.layers_dims)  # seeded to 3

    def run(self, learning_rate):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(self.X, self.parameters)
        cost = compute_cost(a3, self.Y)
        grads = backward_propagation(self.X, self.Y, cache)
        self.parameters = update_parameters(self.parameters, grads, learning_rate)
        return grads, cost

    def train(self, x_train, y_train, *,
              num_iterations=30_000,
              learning_rate=0.3,
              print_cost=True):

        self.prepare(x_train, y_train)
        costs = []
        for i in range(num_iterations):
            _, cost = self.run(learning_rate)

            if i % 1000 == 0:
                costs.append(cost)

            if print_cost and i % 10000 == 0:
                print(f"Cost after iteration {i}: {cost}")

        return costs


class NNModel:

    def __init__(self, features_count, lambd=0):
        hidden_layers_sizes = [20, 3]
        self.weight_initializer = init.XavierInitializer()

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
        self.weight_matrices = []
        for l, h in enumerate(hidden_layers_sizes):
            linear, w, b = self.fully_connected_layer(a, a_size, h, l)
            self.weight_matrices.append(w)
            a = f.relu(linear)
            a_size = h

        # Output layer
        linear, w, b = self.fully_connected_layer(a, a_size, 1, self.L)
        self.weight_matrices.append(w)
        loss = f.sigmoid_cross_entropy(linear, self.Y)
        if lambd > 0:
            loss += reg.l2_norm_regularizer(lambd, nf.dim(self.X), self.weight_matrices)
        predictions = f.is_greater_than(f.sigmoid(linear), 0.5)

        # Computation graphs
        self.cost_graph = g.Graph(loss)
        self.prediction_graph = g.Graph(predictions)

    def fully_connected_layer(self, features, cnt_features, cnt_neurons, layer_num):
        weights = f.var("W" + str(layer_num + 1), self.weight_initializer, shape=(cnt_neurons, cnt_features))
        biases = f.var("b" + str(layer_num + 1), init.ZeroInitializer(), shape=(cnt_neurons, 1))
        return fc.fully_connected(features, weights, biases, first_layer=(layer_num == 0)), weights, biases

    def train(self, x_train, y_train, *,
              num_iterations=30_000,
              learning_rate=0.3,
              print_cost=True):
        init.seed(3)
        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(self.cost_graph, learning_rate=learning_rate)
        costs = []
        for i in range(num_iterations):
            optimizer.run()

            if i % 1000 == 0:
                costs.append(optimizer.cost)

            if print_cost and i % 10000 == 0:
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


def plot_boundary(reg_name, m, xt, yt):
    plt.title(f"Model with regularizer: {reg_name}")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: m.predict(x.T), xt, yt)


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load image dataset: blue/red dots in circles
    train_x, train_y, test_x, test_y = load_2D_dataset()
    # plt.show()

    m_train = train_x.shape[1]
    m_test = test_x.shape[1]
    n = train_x.shape[0]

    # Model
    pr = cProfile.Profile()
    model_name = "mine"
    model = NNModel(n, lambd=0)

    # model_name = "coursera"
    # model = CourseraModel(n, m_train)

    start_time = time.time()
    # pr.enable()

    costs = model.train(train_x, train_y)

    # pr.disable()
    end_time = time.time()

    # plot_costs(costs)
    #
    # # Predict
    # train_predictions = model.predict(train_x)
    # train_accuracy = accuracy(actual=train_predictions, expected=train_y)
    # print(f"Train accuracy = {train_accuracy}%")
    #
    # test_predictions = model.predict(test_x)
    # test_accuracy = accuracy(actual=test_predictions, expected=test_y)
    # print(f"Test accuracy = {test_accuracy}%")
    #
    # # Plot
    # plot_boundary("none", model, train_x, train_y)

    # Performance report
    print("time = ", end_time - start_time)
    # pr.dump_stats(f"stats_{model_name}.prof")
    # pr.create_stats()
