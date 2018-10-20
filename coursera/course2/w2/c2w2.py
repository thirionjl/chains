# import packages
from collections import OrderedDict

import matplotlib.pyplot as plt

from chains.core import env
from chains.core.optimizers import AdamOptimizer
from chains.core.optimizers import GradientDescentOptimizer, MomentumOptimizer
from chains.front.model import Model
from chains.front.network import Dense, Sequence, ReLu
from chains.front.network import SigmoidBinaryClassifier
from chains.front.training import TrainListener, MiniBatchTraining
from coursera.course2.w2.opt_utils import plot_decision_boundary, load_dataset
from coursera.utils import binary_accuracy, plot_costs


# Note: To get exactly same results as in coursera exercise
# you need to replace  np.random.shuffle(shuffled_range)
# with shuffled_range = np.random.permutation(cnt_examples) in training.py
# but results are similar enough without that change
class CostListener(TrainListener):

    def __init__(self):
        env.seed(3)
        self.costs = []
        self.seed = 10

    def on_start(self):
        self.costs = []

    def on_epoch_start(self, epoch):
        self.seed += 1
        env.seed(self.seed)

    def on_epoch_end(self, epoch, cost):
        if epoch % 100 == 0:
            self.costs.append(cost)

        if epoch % 1000 == 0:
            print(f"Cost after epoch {epoch}: {cost}")

    def on_iteration(self, epoch, num_batch, i, cost):
        pass

    def on_end(self):
        plot_costs(self.costs, unit=100, learning_rate=0.0007)


def model(cnt_features):
    return Model(
        network=Sequence(
            cnt_features=cnt_features,
            layers=[
                Dense(5), ReLu(),
                Dense(2), ReLu(),
                Dense(1),
            ],
            classifier=SigmoidBinaryClassifier(),
        )
    )


def show_image(i, im_classes, x, y):
    plt.imshow(x[i])
    plt.show()
    print("y = " + str(y[0, i]) + ". It's a " + im_classes[y[0, i]].decode(
        "utf-8") + " picture.")


def plot_boundary(reg_name, m, xt, yt):
    plt.title(f"Model with regularizer: {reg_name}")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x: m.predict(x.T), xt, yt)


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # load image dataset: blue/red dots in circles
    train_x, train_y = load_dataset()

    m_train = train_x.shape[1]
    n = train_x.shape[0]

    # Model
    model = model(n)
    optimizers = OrderedDict()
    optimizers["gradient descent"] = GradientDescentOptimizer(0.0007)
    optimizers["momentum"] = MomentumOptimizer(0.0007, 0.9)
    optimizers["adam"] = AdamOptimizer(0.0007, 0.9, 0.999)

    for name, opt in optimizers.items():
        training = MiniBatchTraining(
            batch_size=64,
            optimizer=opt,
            listener=CostListener()
        )

        model.train(train_x, train_y, epochs=10_000, training=training)
        train_predictions = model.predict(train_x)
        train_accuracy = binary_accuracy(train_predictions, train_y)
        print(f"Train accuracy = {train_accuracy}%")
        plot_boundary(name, model, train_x, train_y)
