# import packages
import time

import matplotlib.pyplot as plt
import numpy as np

from chains.core import env
from chains.core.initializers import VarInitializer
from chains.core.optimizers import AdamOptimizer
from chains.core.preprocessing import one_hot
from chains.front.model import Model
from chains.front.network import Dense, Sequence, SoftmaxClassifier, ReLu
from chains.front.training import TrainListener, MiniBatchTraining
from coursera.course2.w3.tf_utils import load_dataset
from coursera.course2.w3.tf_utils import load_initial_parameter_values
from coursera.utils import plot_costs


class H5Init(VarInitializer):

    def initialize(self, param, dtype):
        self.layer_num += 1
        return self.ws[self.layer_num].astype(dtype)

    def __init__(self):
        self.layer_num = -1
        self.ws = load_initial_parameter_values()


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
        if epoch % 1 == 0:
            print(f"Cost after epoch {epoch}: {cost}")

    def on_end(self):
        plot_costs(self.costs, unit=5, learning_rate=0.0001)


Dense.default_weight_initializer = H5Init()


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
    import daz

    daz.unset_daz()
    daz.unset_ftz()
    np.seterr(under='raise')

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

    model.train(train_x.astype(dtype=np.float32),
                train_y.astype(dtype=np.float32), epochs=1_500)
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
