from chains.core.metrics import accuracy
from chains.core.optimizers import AdamOptimizer
from chains.core.preprocessing import one_hot
from chains.front.model import Model
from chains.front.network import BatchNorm
from chains.front.network import Dense, Sequence, SoftmaxClassifier, ReLu
from chains.front.training import MiniBatchTraining
from chains.tools import text_exporter, graphviz_exporter
from chains.tools.backup import restore_network
from coursera.course2.w3.c2w3 import CostListener
from coursera.course2.w3.tf_utils import load_dataset


def model(cnt_features):
    return Model(
        network=Sequence(
            cnt_features=cnt_features,
            layers=[
                Dense(25),
                BatchNorm(),
                ReLu(),
                Dense(12),
                BatchNorm(),
                ReLu(),
                Dense(6),
            ],
            classifier=SoftmaxClassifier(classes=6),
        ),
        training=MiniBatchTraining(
            batch_size=32, optimizer=AdamOptimizer(0.0001), listener=CostListener()
        ),
    )


if __name__ == "__main__":
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_dataset()

    # Pre-processing
    train_x_flat = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flat = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    train_x = train_x_flat / 255.0
    test_x = test_x_flat / 255.0
    train_y = one_hot(train_y_orig, 6)
    test_y = one_hot(test_y_orig, 6)
    n = train_x.shape[0]

    # Model
    model = model(n)

    # Restore
    restore_network(model.network, "datasets/c2w3_trained_weights.hdf5")

    # Check accuracy
    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)
    print(f"Train accuracy = {accuracy(train_y_orig, train_predictions)}%")
    print(f"Test accuracy = {accuracy(test_y_orig, test_predictions)}%")

    # Drawing prep
    gz = graphviz_exporter.export(model.network.cost_graph)
    gz.view()
    print(text_exporter.export(model.network.cost_graph))
