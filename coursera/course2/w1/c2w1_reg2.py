import time

import matplotlib.pyplot as plt

import coursera.course2.w1.reg_utils as utils
from chains.core import env, metrics
from chains.core.initializers import XavierInitializer
from chains.core.optimizers import GradientDescentOptimizer
from chains.front.model import Model
from chains.front.network import Dense, Sequence, ReLu, Dropout
from chains.front.network import SigmoidBinaryClassifier, L2Regularizer
from chains.front.training import TrainListener, BatchTraining
from chains.tools import text_exporter, graphviz_exporter
from coursera.utils import plot_costs

Dense.default_weight_initializer = XavierInitializer()


class CostListener(TrainListener):
    def __init__(self):
        self.costs = []
        self.start_time = None

    def on_start(self):
        self.costs = []
        self.start_time = time.time()
        env.seed(3)

    def on_epoch_start(self, epoch_num):
        env.seed(1)

    def on_iteration(self, epoch, i, cost):
        if i % 1000 == 0:
            self.costs.append(cost)

        if i % 10000 == 0:
            print(f"Cost after iteration {i}: {cost}")

    def on_end(self):
        print("time = ", time.time() - start_time)
        plot_costs(self.costs, unit=1000, learning_rate=0.3)


batch_gd = BatchTraining(GradientDescentOptimizer(0.3), CostListener())


def default_model(cnt_features):
    return Model(
        network=Sequence(
            cnt_features=cnt_features,
            layers=[
                Dense(20, namespace="layers.1"),
                ReLu(namespace="layers.1"),
                Dense(3, namespace="layers.2"),
                ReLu(namespace="layers.2"),
                Dense(1, namespace="layers.3"),
            ],
            classifier=SigmoidBinaryClassifier(),
        ),
        training=batch_gd,
    )


def l2reg_model(cnt_features):
    return Model(
        network=Sequence(
            cnt_features=cnt_features,
            layers=[
                Dense(20, namespace="1"),
                ReLu(namespace="1"),
                Dense(3, namespace="2"),
                ReLu(namespace="2"),
                Dense(1, namespace="3"),
            ],
            classifier=SigmoidBinaryClassifier(),
            regularizer=L2Regularizer(lambd=0.7),
        ),
        training=batch_gd,
    )


def dropout_model(cnt_features):
    return Model(
        network=Sequence(
            cnt_features=cnt_features,
            layers=[
                Dense(20, namespace="1"),
                ReLu(namespace="1"),
                Dropout(0.86, namespace="1"),
                Dense(3, namespace="2"),
                ReLu(namespace="2"),
                Dropout(0.86, namespace="2"),
                Dense(1, namespace="3"),
            ],
            classifier=SigmoidBinaryClassifier(),
        ),
        training=batch_gd,
    )


def show_image(i, im_classes, x, y):
    plt.imshow(x[i])
    plt.show()
    print(
        "y = "
        + str(y[0, i])
        + ". It's a "
        + im_classes[y[0, i]].decode("utf-8")
        + " picture."
    )


def plot_boundary(reg_name, m, xt, yt):
    plt.title(f"Model with regularizer: {reg_name}")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    utils.plot_decision_boundary(lambda x: m.predict(x.T), xt, yt)


def display_model(md: Model):
    gz = graphviz_exporter.export(md.network.cost_graph)
    gz.view()
    print(text_exporter.export(md.network.cost_graph))


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (7.0, 4.0)  # set default size of plots
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "gray"

    # load image dataset: blue/red dots in circles
    train_x, train_y, test_x, test_y = utils.load_2D_dataset()

    m_train = train_x.shape[1]
    m_test = test_x.shape[1]
    n = train_x.shape[0]

    # Model
    models = [default_model(n), l2reg_model(n), dropout_model(n)]

    for model in models:
        utils.display_graph(model.network.cost_graph)
        input("Press enter to continue: ")

    for model in models:
        # Train
        train_x = train_x.astype("float32")
        start_time = time.time()
        model.train(train_x, train_y, epochs=30_000)

        # Predict
        train_predictions = model.predict(train_x)
        train_accuracy = metrics.accuracy(train_predictions, train_y)
        print(f"Train accuracy = {train_accuracy}%")

        test_predictions = model.predict(test_x)
        test_accuracy = metrics.accuracy(test_predictions, test_y)
        print(f"Test accuracy = {test_accuracy}%")

        # Plot
        plot_boundary("none", model, train_x, train_y)
