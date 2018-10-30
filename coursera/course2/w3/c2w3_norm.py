# import packages

import matplotlib.pyplot as plt
import numpy as np

from chains.core.metrics import accuracy
from chains.core.optimizers import AdamOptimizer
from chains.core.preprocessing import one_hot
from chains.front.model import Model
from chains.front.network import BatchNorm
from chains.front.network import Dense, Sequence, SoftmaxClassifier, ReLu
from chains.front.training import MiniBatchTraining
from coursera.course2.w3.c2w3 import H5Init, CostListener
from coursera.course2.w3.tf_utils import load_dataset

Dense.default_weight_initializer = H5Init()


def model(cnt_features):
    return Model(
        network=Sequence(
            cnt_features=cnt_features,
            layers=[
                Dense(25), BatchNorm(), ReLu(),
                Dense(12), BatchNorm(), ReLu(),
                Dense(6),
            ],
            classifier=SoftmaxClassifier(classes=6),
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

    # Train
    model.train(train_x.astype(dtype=np.float32),
                train_y.astype(dtype=np.int16), epochs=1_000)

    # Check accuracy
    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)
    print(f"Train accuracy = {accuracy(train_y_orig, train_predictions)}%")
    print(f"Test accuracy = {accuracy(test_y_orig, test_predictions)}%")
