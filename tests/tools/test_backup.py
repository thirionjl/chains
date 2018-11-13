import operator
import os

import h5py
import numpy as np
from numpy.testing import assert_allclose

from chains.core.optimizers import AdamOptimizer
from chains.core.preprocessing import one_hot
from chains.front.model import Model
from chains.front.network import Sequence, Dense, BatchNorm, ReLu, \
    SoftmaxClassifier
from chains.front.training import MiniBatchTraining, TrainListener
from chains.tools.backup import save_network, restore_network

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_backup_network(tmpdir):
    test_x_orig, test_y_orig, classes = load_dataset()
    test_x_flat = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    test_x = test_x_flat / 255.
    test_y = one_hot(test_y_orig, 6)
    n = test_x.shape[0]

    model = Model(
        network=Sequence(
            cnt_features=n,
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
            listener=TrainListener()
        )
    )

    # Train
    model.train(test_x.astype(dtype=np.float32),
                test_y.astype(dtype=np.int16), epochs=5)

    path = tmpdir.mkdir("test_backup").join("bak.hdf5")

    save_network(model.network, path)

    model2 = Model(
        network=Sequence(
            cnt_features=n,
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
            listener=TrainListener()
        )
    )

    restore_network(model2.network, path)

    vars1 = variables(model)
    vars2 = variables(model2)

    for kv1, kv2 in zip(vars1, vars2):
        assert kv1[0] == kv2[0]
        assert_allclose(kv1[1], kv2[1])


def variables(model):
    vs = {v.name: v.value for v in model.network.cost_graph.variables}
    return sorted(vs.items(), key=operator.itemgetter(0))


def load_dataset():
    with h5py.File(os.path.join(dir_path, 'test_signs.h5'), "r") as ds:
        test_x_orig = np.array(ds["test_set_x"][:])
        test_y_orig = np.array(ds["test_set_y"][:])
        classes = np.array(ds["list_classes"][:])
        test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))

    return test_x_orig, test_y_orig, classes
