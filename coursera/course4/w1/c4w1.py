# import packages
import time

import matplotlib.pyplot as plt
import numpy as np

from chains.core import env
from chains.core.initializers import XavierInitializer
from chains.core.metrics import accuracy
from chains.core.optimizers import AdamOptimizer
from chains.core.preprocessing import one_hot
from chains.front.model import Model
from chains.front.network import Dense, Sequence, SoftmaxClassifier, ReLu, \
    Conv2dNoBias, MaxPool, Flatten, Transpose
from chains.front.training import TrainListener, MiniBatchTraining
from coursera.course4.w1.tf_utils import load_dataset
from coursera.utils import plot_costs


class CostListener(TrainListener):

    def __init__(self):
        self.costs = []
        self.seed = None
        self.start_time = None

    def on_start(self):
        self.seed = 0
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


Dense.default_weight_initializer = XavierInitializer()
Conv2dNoBias.default_weight_initializer = XavierInitializer()


def model():
    return Model(
        network=Sequence(
            feature_shape=(3, 64, 64),
            sample_axis_last=False,
            layers=[
                Conv2dNoBias(fh=4, fw=4, channels=8, padding=2), ReLu(),
                MaxPool(stride=8),  # TODO PAdding !!!
                Conv2dNoBias(fh=2, fw=2, channels=16, padding=1), ReLu(),
                MaxPool(stride=4),  # TODO PAdding !!!
                Transpose(axes=(1, 2, 3, 0)),
                Flatten(),
                Dense(6),
            ],
            classifier=SoftmaxClassifier(classes=6),
        ),
        training=MiniBatchTraining(
            batch_size=64,
            optimizer=AdamOptimizer(0.009),
            listener=CostListener()
        )
    )


def show_image(i, x, y):
    plt.imshow(x[i])
    plt.show()
    print("y = " + str(np.squeeze(y[:, i])))


if __name__ == "__main__":
    import daz

    daz.set_ftz()
    np.seterr(under='warn')

    plt.rcParams['figure.figsize'] = (7.0, 4.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load image dataset: blue/red dots in circles
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = \
        load_dataset()
    # show_image(0, train_x_orig, train_y_orig)

    # Pre-processing
    train_x = train_x_orig.transpose(0, 3, 1, 2) / 255.
    test_x = test_x_orig.transpose(0, 3, 1, 2) / 255.
    train_y = one_hot(train_y_orig, 6)
    test_y = one_hot(test_y_orig, 6)
    m_train = train_x.shape[1]

    show_image(6, train_x_orig, train_y_orig)

    # Model
    model = model()

    # Train
    model.train(train_x.astype(dtype=np.float32),
                train_y.astype(dtype=np.int16), epochs=100)

    # Check accuracy
    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)
    print(f"Train accuracy = {accuracy(train_y_orig, train_predictions)}%")
    print(f"Test accuracy = {accuracy(test_y_orig, test_predictions)}%")
