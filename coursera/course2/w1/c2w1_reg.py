# import packages
import cProfile
import time

from chains.core import metrics as m
from chains.core import node_factory as f, initializers as init
from chains.core import optimizers as gd, graph as g, env
from chains.core.shape import Dim
from coursera.course2.w1.reg_utils import *
from coursera.utils import plot_costs

ITERATION_UNIT = 1_000


class NNModel:

    def __init__(self, features_count, *, lambd=0, keep_prob=1):
        hidden_layers_sizes = [20, 3]

        self.m = Dim.unknown()
        self.n = features_count
        self.L = len(hidden_layers_sizes)

        self.X = f.placeholder(shape=(self.n, self.m), dtype=np.float64)
        self.Y = f.placeholder(shape=(1, self.m))
        weights, biases = self.build_variables(self.n, hidden_layers_sizes)

        # Cost graph
        network = self.layers(self.X, weights, biases, self.L, keep_prob)
        loss = f.sigmoid_cross_entropy(network, self.Y)
        if lambd > 0:
            loss += f.l2_norm_regularizer(lambd, f.dim(self.X), weights)
        self.cost_graph = g.Graph(loss)

        # Prediction graph
        network = self.layers(self.X, weights, biases, self.L, 1)
        predictions = f.is_greater_than(f.sigmoid(network), 0.5)
        self.prediction_graph = g.Graph(predictions)

    @staticmethod
    def layers(features_matrix, weight_matrices, bias_matrices,
               layers_count, keep_prob=1):
        a = features_matrix
        weights_and_bias = list(zip(weight_matrices, bias_matrices))
        for l, (w, b) in enumerate(weights_and_bias[:-1]):
            linear = NNModel.layer(a, w, b, l + 1)
            a = f.relu(linear)

            if 0 < keep_prob < 1:
                a = f.dropout(keep_prob, a)

        return NNModel.layer(a, weight_matrices[-1], bias_matrices[-1],
                             layers_count)

    @staticmethod
    def build_variables(features_count, hidden_layers_sizes):
        a_size = features_count
        weight_matrices = []
        bias_matrices = []
        i = 0
        for i, h in enumerate(hidden_layers_sizes):
            w = f.var("W" + str(i + 1), init.XavierInitializer(),
                      shape=(h, a_size))
            b = f.var("b" + str(i + 1), init.ZeroInitializer(), shape=(h, 1))
            weight_matrices.append(w)
            bias_matrices.append(b)
            a_size = h

        w = f.var("W" + str(i + 2), init.XavierInitializer(),
                  shape=(1, a_size))
        b = f.var("b" + str(i + 2), init.ZeroInitializer(), shape=(1, 1))
        weight_matrices.append(w)
        bias_matrices.append(b)
        return weight_matrices, bias_matrices

    @staticmethod
    def layer(features, w, b, layer_num):
        return f.fully_connected(features, w, b, first_layer=(layer_num == 1))

    def train(self, x_train, y_train, *,
              num_iterations=30_000,
              learning_rate=0.3,
              print_cost=True):
        env.seed(3)
        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(learning_rate)
        optimizer.initialize_and_check(self.cost_graph)
        costs = []
        for i in range(num_iterations):
            env.seed(1)
            optimizer.run()

            if i % ITERATION_UNIT == 0:
                costs.append(optimizer.cost)

            if print_cost and i % 10000 == 0:
                print(f"Cost after iteration {i}: {optimizer.cost}")

        return costs

    def predict(self, x_test):
        self.prediction_graph.placeholders = {self.X: x_test}
        return self.prediction_graph.evaluate()


def show_image(i, im_classes, x, y):
    plt.imshow(x[i])
    plt.show()
    print("y = " + str(y[0, i]) + ". It's a " + im_classes[y[0, i]].decode(
        "utf-8") + " picture.")


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

    m_train = train_x.shape[1]
    m_test = test_x.shape[1]
    n = train_x.shape[0]

    # Model
    pr = cProfile.Profile()
    models = [NNModel(n), NNModel(n, lambd=0.7), NNModel(n, keep_prob=0.86)]

    for model in models:
        # Train
        start_time = time.time()
        costs = model.train(train_x, train_y)
        end_time = time.time()

        plot_costs(costs, unit=ITERATION_UNIT, learning_rate=0.3)

        # Predict
        train_predictions = model.predict(train_x)
        train_accuracy = m.accuracy(train_predictions, train_y)
        print(f"Train accuracy = {train_accuracy}%")

        test_predictions = model.predict(test_x)
        test_accuracy = m.accuracy(test_predictions, test_y)
        print(f"Test accuracy = {test_accuracy}%")

        # Plot
        plot_boundary("none", model, train_x, train_y)

        # Performance report
        print("time = ", end_time - start_time)
