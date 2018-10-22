# import packages
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from chains.core import env
from chains.core.initializers import VarInitializer
from chains.core.optimizers import AdamOptimizer
from chains.core.preprocessing import one_hot
from chains.front.model import Model
from chains.front.network import Dense, Sequence, SoftmaxClassifier, ReLu
from chains.front.training import TrainListener, MiniBatchTraining
from coursera.course2.w3.tf_utils import load_dataset
from coursera.utils import plot_costs


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25, 12288],
                         initializer=tf.contrib.layers.xavier_initializer(
                             seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25],
                         initializer=tf.contrib.layers.xavier_initializer(
                             seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12],
                         initializer=tf.contrib.layers.xavier_initializer(
                             seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


class TensorflowInit(VarInitializer):

    def initialize(self, param, dtype):
        self.layer_num += 1
        return self.ws[self.layer_num]

    def __init__(self):
        with tf.Session() as sess:
            tf.set_random_seed(1)
            parameters = initialize_parameters()
            init = tf.global_variables_initializer()
            sess.run(init)
            w1 = sess.run(parameters["W1"])
            w2 = sess.run(parameters["W2"])
            w3 = sess.run(parameters["W3"])
            self.ws = [np.array(w1).astype(dtype=np.float32),
                       np.array(w2).astype(dtype=np.float32),
                       np.array(w3).astype(dtype=np.float32)]
        self.layer_num = -1


class CostListener(TrainListener):

    def __init__(self):
        self.costs = []
        self.seed = None
        self.start_time = None

    def on_start(self):
        self.seed = 3
        self.costs = []

    def on_epoch_start(self, epoch_num):
        self.seed += 1
        env.seed(self.seed)
        if epoch_num % 100 == 0:
            if epoch_num > 0:
                duration = time.time() - self.start_time
                print(f"100 epoch duration = {duration}")
            self.start_time = time.time()

    def on_epoch_end(self, epoch, cost):
        if epoch % 5 == 0:
            self.costs.append(cost)
        if epoch % 100 == 0:
            print(f"Cost after epoch {epoch}: {cost}")

    def on_end(self):
        plot_costs(self.costs, unit=5, learning_rate=0.0001)


Dense.default_weight_initializer = TensorflowInit()


def model(cnt_features):
    return Model(
        network=Sequence(
            cnt_features=cnt_features,
            cnt_labels=6,
            layers=[
                Dense(25), ReLu(),
                Dense(12), ReLu(),
                Dense(6),
            ],
            classifier=SoftmaxClassifier(),
        ),
        training=MiniBatchTraining(
            batch_size=32,
            optimizer=AdamOptimizer(0.0001),
            listener=CostListener()
        )
    )


def show_image(i, x, y):
    plt.imshow(x[i])
    plt.show()
    print("y = " + str(np.squeeze(y[:, i])))

    # def plot_boundary(reg_name, m, xt, yt):
    #     plt.title(f"Model with regularizer: {reg_name}")
    #     axes = plt.gca()
    #     axes.set_xlim([-1.5, 2.5])
    #     axes.set_ylim([-1, 1.5])
    #     plot_decision_boundary(lambda x: m.predict(x.T), xt, yt)


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (7.0, 4.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load image dataset: blue/red dots in circles
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = \
        load_dataset()

    # show_image(0, train_x_orig, train_y_orig)

    # Preprocessing
    train_x_flat = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flat = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    train_x = train_x_flat / 255.
    test_x = test_x_flat / 255.
    train_y = one_hot(train_y_orig, 6)
    test_y = one_hot(test_y_orig, 6)

    m_train = train_x.shape[1]
    n = train_x.shape[0]

    # Model
    model = model(n)

    model.train(train_x.astype(dtype=np.float32), train_y.astype(dtype=np.float32), epochs=1_500)
    train_predictions = model.predict(train_x)
    correct_prediction = np.equal(train_y_orig, train_predictions)
    train_accuracy = np.mean(correct_prediction) * 100

    print(f"Train accuracy = {train_accuracy}%")
    # plot_boundary(name, model, train_x, train_y)

    # optimizers = OrderedDict()
    # optimizers["gradient descent"] = GradientDescentOptimizer(0.0007)
    # optimizers["momentum"] = MomentumOptimizer(0.0007, 0.9)
    # optimizers["adam"] = AdamOptimizer(0.0007, 0.9, 0.999)
    #
    # for name, opt in optimizers.items():
    #     training = MiniBatchTraining(
    #         batch_size=64,
    #         optimizer=opt,
    #         listener=CostListener()
    #     )
    #
    #     model.train(train_x, train_y, epochs=10_000, training=training)
    #     train_predictions = model.predict(train_x)
    #     train_accuracy = binary_accuracy(train_predictions, train_y)
    #     print(f"Train accuracy = {train_accuracy}%")
    #     plot_boundary(name, model, train_x, train_y)
