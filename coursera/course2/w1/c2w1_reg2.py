# import packages
import time

from chains.model.model import *
from coursera.course2.w1.reg_utils import *
from coursera.utils import binary_accuracy, plot_costs


def default_model(cnt_features):
    return Sequence(
        cnt_features=cnt_features,
        layers=[
            FullyConnectedLayer(20, weight_initializer=init.XavierInitializer()),
            ReLuLayer(),
            FullyConnectedLayer(3, weight_initializer=init.XavierInitializer()),
            ReLuLayer(),
            FullyConnectedLayer(1, weight_initializer=init.XavierInitializer()),
        ],
        classifier=SigmoidBinaryClassifier(),
    )


def l2reg_model(cnt_features):
    return Sequence(
        cnt_features=cnt_features,
        layers=[
            FullyConnectedLayer(20, weight_initializer=init.XavierInitializer()),
            ReLuLayer(),
            FullyConnectedLayer(3, weight_initializer=init.XavierInitializer()),
            ReLuLayer(),
            FullyConnectedLayer(1, weight_initializer=init.XavierInitializer()),
        ],
        classifier=SigmoidBinaryClassifier(),
        regularizer=L2Regularizer(lambd=0.7)
    )


def dropout_model(cnt_features):
    return Sequence(
        cnt_features=cnt_features,
        layers=[
            FullyConnectedLayer(20, weight_initializer=init.XavierInitializer()),
            ReLuLayer(),
            DropoutLayer(0.86),
            FullyConnectedLayer(3, weight_initializer=init.XavierInitializer()),
            ReLuLayer(),
            DropoutLayer(0.86),
            FullyConnectedLayer(1, weight_initializer=init.XavierInitializer()),
        ],
        classifier=SigmoidBinaryClassifier(),
    )


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
    models = [default_model(n), l2reg_model(n), dropout_model(n)]

    for model in models:
        # Train
        start_time = time.time()
        costs = model.train(train_x, train_y, num_iterations=30_000, learning_rate=0.3, print_cost=True)
        end_time = time.time()

        plot_costs(costs, unit=1000, learning_rate=0.3)

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
        plot_boundary("none", model, train_x, train_y)

        # Performance report
        print("time = ", end_time - start_time)
