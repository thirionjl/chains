import h5py
import numpy as np


def load_dataset():
    with h5py.File('datasets/train_signs.h5', "r") as train_ds:
        train_x_orig = np.array(train_ds["train_set_x"][:])
        train_y_orig = np.array(train_ds["train_set_y"][:])
        train_y_orig = train_y_orig.reshape((1, train_y_orig.shape[0]))

    with h5py.File('datasets/test_signs.h5', "r") as test_dataset:
        test_x_orig = np.array(test_dataset["test_set_x"][:])
        test_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])
        test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))

    return train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes


# def load_initial_parameter_values():
#     with h5py.File("datasets/initializations_signs.hdf5", "r") as f:
#         return [np.array(f["W1"]), np.array(f["W2"]), np.array(f["W3"])]
