import matplotlib.pyplot as plt

from chains.graph import node_factory as f, structure as g
from chains.initialization import variable_initializers as init
from chains.layers import fully_connected as fc
from chains.optimizer import gradient_descent as gd
from chains.tensor.tensor import Dim
from chains import env
from coursera.course1.w3.planar_utils import plot_decision_boundary, \
    load_planar_dataset
from coursera.utils import binary_accuracy

ITERATION_UNIT = 100


class ShallowNNModel:

    def __init__(self, features_count, hidden_layers_size):
        # Number of examples
        self.m = Dim.unknown()
        # Number of features
        self.n = features_count
        # Number of hidden units
        self.h = hidden_layers_size
        weight_initializer = init.RandomNormalInitializer()
        bias_initializer = init.ZeroInitializer()

        # Placeholders and Vars
        self.W1 = f.var("W1", weight_initializer, shape=(self.h, self.n))
        self.b1 = f.var("b1", bias_initializer, shape=(self.h, 1))
        self.W2 = f.var("W2", weight_initializer, shape=(1, self.h))
        self.b2 = f.var("b2", bias_initializer, shape=(1, 1))
        self.X = f.placeholder(shape=(self.n, self.m))
        self.Y = f.placeholder(shape=(1, self.m))

        # Nodes
        lin_1 = fc.fully_connected(self.X, self.W1, self.b1, first_layer=True)
        act_1 = f.tanh(lin_1)
        lin_2 = fc.fully_connected(act_1, self.W2, self.b2)
        loss = f.sigmoid_cross_entropy(lin_2, self.Y)
        predictions = f.is_greater_than(f.sigmoid(lin_2), 0.5)

        # Computation graphs
        self.cost_graph = g.Graph(loss)
        self.prediction_graph = g.Graph(predictions)

    def train(self, x_train, y_train, num_iterations=10_000, learning_rate=1.2,
              print_cost=False):
        env.seed(3)
        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(self.cost_graph,
                                                learning_rate=learning_rate)
        costs = []
        for i in range(num_iterations):
            optimizer.run()

            if i % ITERATION_UNIT == 0:
                costs.append(optimizer.cost)

            if print_cost and i % ITERATION_UNIT == 0:
                print(f"Cost after iteration {i}: {optimizer.cost}")

        return costs

    def predict(self, x_test):
        self.prediction_graph.placeholders = {self.X: x_test}
        return self.prediction_graph.evaluate()


if __name__ == "__main__":
    # Dataset Loading
    X, Y = load_planar_dataset()
    shape_X = X.shape
    shape_Y = Y.shape
    m = shape_X[1]  # training set size

    # Test different hidden layer sizes
    plt.figure()
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, h in enumerate(hidden_layer_sizes):
        print(f"\n\n>>> Testing with hidden layer of size {h}")

        # Model
        model = ShallowNNModel(X.shape[0], h)
        model.train(X, Y, num_iterations=5_000)

        # Predict
        train_predictions = model.predict(X)
        train_accuracy = binary_accuracy(actual=train_predictions, expected=Y)
        print(f"Train accuracy = {train_accuracy}%")

        # Plot
        plt.subplot(5, 2, i + 1)
        plt.title(f"Hidden Layer of size {h}")
        plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
        plt.title(f"Decision Boundary for hidden layer size {h}")

    plt.show()
