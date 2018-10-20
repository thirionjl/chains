import matplotlib.pyplot as plt

from chains.core import node_factory as f, initializers as init
from chains.core import optimizers as gd, graph as g, env
from chains.core.shape import Dim
from coursera.course2.w1.init_utils import load_dataset, plot_decision_boundary
from coursera.utils import binary_accuracy, plot_costs

ITERATION_UNIT = 1000


class NNModel:
    initializers = {
        "zeros": init.ZeroInitializer(),
        "random": init.RandomNormalInitializer(20),
        "he": init.HeInitializer(),
    }

    def __init__(self, features_count, initializer_name,
                 hidden_layers_sizes=[10, 5]):
        self.weight_initializer = self.initializers.get(initializer_name)
        # Number of examples
        self.m = Dim.unknown()
        # Number of features
        self.n = features_count
        # Number of hidden units
        self.L = len(hidden_layers_sizes)

        # Placeholders and Vars
        self.X = f.placeholder(shape=(self.n, self.m))
        self.Y = f.placeholder(shape=(1, self.m))

        # Hidden todo
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

    def fully_connected_layer(self, features, cnt_features, cnt_neurons,
                              layer_num):
        weights = f.var("W" + str(layer_num + 1), self.weight_initializer,
                        shape=(cnt_neurons, cnt_features))
        biases = f.var("b" + str(layer_num + 1), init.ZeroInitializer(),
                       shape=(cnt_neurons, 1))
        return f.fully_connected(features, weights, biases,
                                 first_layer=(layer_num == 0))

    def train(self, x_train, y_train, *,
              num_iterations=15_000,
              learning_rate=0.01,
              print_cost=True):

        env.seed(3)
        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(learning_rate)
        optimizer.initialize_and_check(self.cost_graph)
        costs = []
        for i in range(num_iterations + 1):
            optimizer.run()

            if i % 1000 == 0:
                costs.append(optimizer.cost)

            if print_cost and i % 1000 == 0:
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

    for initializer in NNModel.initializers.keys():
        # Model
        model = NNModel(n, initializer)
        costs = model.train(train_x, train_y, print_cost=True)
        plot_costs(costs, unit=ITERATION_UNIT, learning_rate=0.01)

        # Predict
        train_predictions = model.predict(train_x)
        train_accuracy = binary_accuracy(actual=train_predictions,
                                         expected=train_y)
        print(f"Train accuracy = {train_accuracy}%")

        test_predictions = model.predict(test_x)
        test_accuracy = binary_accuracy(actual=test_predictions,
                                        expected=test_y)
        print(f"Test accuracy = {test_accuracy}%")

        # Plot
        plot_boundary(initializer, model, train_x, train_y)
