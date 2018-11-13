import time

import matplotlib.pyplot as plt
import numpy as np

import coursera.course1.w2.lr_utils as cs
from chains.core import metrics as m
from chains.core import node_factory as f, initializers as init
from chains.core import node_factory as nf
from chains.core import optimizers as gd, graph as g
from chains.core.shape import Dim
from coursera.utils import plot_costs

ITERATIONS_UNIT = 100


class LogisticRegressionModel:

    def __init__(self, features_count):

        self.m = Dim.unknown()  # Number of examples
        self.n = features_count  # Number of features

        self.W = f.var('W', init.ZeroInitializer(), shape=(1, self.n))
        self.b = f.var('b', init.ZeroInitializer(), shape=(1, 1))
        self.X = f.placeholder(shape=(self.n, self.m))
        self.Y = f.placeholder(shape=(1, self.m))

        self.logits = nf.fully_connected(self.X, self.W, self.b,
                                         first_layer=True)
        self.loss = f.sigmoid_cross_entropy(self.logits, self.Y)
        self.predictions = f.is_greater_than(f.sigmoid(self.logits), 0.5)

        self.cost_graph = g.Graph(self.loss)
        self.prediction_graph = g.Graph(self.predictions)

    def train(self, x_train, y_train, num_iterations=2000, learning_rate=0.5,
              print_cost=True):
        self.cost_graph.placeholders = {self.X: x_train, self.Y: y_train}
        self.cost_graph.initialize_variables()
        optimizer = gd.GradientDescentOptimizer(learning_rate)
        optimizer.prepare_and_check(self.cost_graph)
        costs = []
        for i in range(num_iterations):
            optimizer.run()

            if i % ITERATIONS_UNIT == 0:
                costs.append(optimizer.cost)

            if print_cost and i % ITERATIONS_UNIT == 0:
                print(f"Cost after iteration {i}: {optimizer.cost}")

        return costs

    def predict(self, x_test):
        self.prediction_graph.placeholders = {self.X: x_test}
        return self.prediction_graph.evaluate()


if __name__ == "__main__":
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = cs.load_dataset()

    print("===== Data set exploration =====")
    print("train_set_x_orig.shape=", train_set_x_orig.shape)
    print("test_set_x_orig.shape=", test_set_x_orig.shape)
    print("train_set_y.shape=", train_set_y.shape)
    print("test_set_y.shape=", test_set_y.shape)
    print("classes=", classes)

    index = 11
    im = plt.imshow(train_set_x_orig[index])
    plt.show()
    label = train_set_y[:, index]  # It's a row vector
    label_class = classes[np.asscalar(label)].decode("utf-8")
    print(f"y = {label}, it's a '{label_class}' picture.")

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    print("Number of training examples: m_train = ", m_train)
    print("Number of testing examples: m_test = ", m_test)
    print("Height/Width of each image: num_px = ", num_px)
    print(f"Each image is of size: ({num_px}, {num_px}, 3)")

    print("==== Pre-processing ====")

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],
                                                   -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],
                                                 -1).T

    print("train_set_x_flatten shape: ", train_set_x_flatten.shape)
    print("train_set_y shape: ", train_set_y.shape)
    print("test_set_x_flatten shape: ", test_set_x_flatten.shape)
    print("test_set_y shape: ", test_set_y.shape)

    # Normalize
    train_set_x = np.array(train_set_x_flatten / 255., dtype=np.float32)
    test_set_x = np.array(test_set_x_flatten / 255, dtype=np.float32)

    print("=== Train ===")
    pixels = num_px * num_px * 3
    model = LogisticRegressionModel(pixels)
    lr = 0.005

    train_set_y = train_set_y.astype(np.float32)

    start_time = time.time()
    costs = model.train(train_set_x, train_set_y, learning_rate=lr,
                        num_iterations=2000, print_cost=True)
    print("training time = ", time.time() - start_time)

    plot_costs(costs, unit=ITERATIONS_UNIT, learning_rate=lr)

    # Predict
    train_predictions = model.predict(train_set_x)
    test_predictions = model.predict(test_set_x)
    print("Train accuracy = ", m.accuracy(train_predictions, train_set_y), "%")
    print("Test  accuracy = ", m.accuracy(test_predictions, test_set_y), "%")

    # Analyze
    # Example of a picture that was wrongly classified.
    errors = (index for index, (p, r) in
              enumerate(zip(test_predictions[0, :], test_set_y[0, :])) if
              p != r)

    index = next(errors)

    label_prediction = test_set_y[0, index]
    class_prediction = classes[int(test_predictions[0, index])].decode('utf-8')
    print(
        f"y = {label_prediction}, you predicted that it is a "
        f"{class_prediction} picture.")

    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    plt.show()
