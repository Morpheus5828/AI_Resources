import numpy as np
from pickle import load, dump
from pathlib import Path


def one_hot_encoding(x: np.array) -> np.array:
    """
    Given an array of integers, this function returns the one-hot encoding of each label as an array of arrays.

    :param x: an array of integers
    :return: an array of one-hot encoded arrays
    """

    one_hot_encoded = np.zeros((x.size, x.max() + 1))
    one_hot_encoded[np.arange(x.size), x] = 1
    return one_hot_encoded

def generate_batches(x: np.array, y: np.array, batch_size: int) -> iter:
    """
    This method creates a generator of randomized batches of size batch_size.

    :param x: an array of input vectors
    :param y: an array of label vectors
    :param batch_size: desired batch_size
    :return: a generator with shuffled data
    """
    while True:
        indices = np.arange(0, x.shape[0], 1)
        np.random.shuffle(indices)
        indices = indices[:batch_size]
        yield (x[indices], y[indices])
        

def load_network(path: str):
    """
    Given a path relative to the "main.py" file, this function loads and returns a neural network from the file found at
    "path" as a NeuralNetwork instance.

    :param path: path to the binary file where the network to be loaded is stored.
    :return: the NeuralNetwork instance stored in the file found at "path"
    """

    path = Path(__file__).parent.joinpath(path).with_suffix(".nn")

    with path.open('rb') as file:
        return load(file)

def save_network(neural_network, path: str) -> None:
    """
    Saves the neural network as a binary .ai file to the path indicated (relative to the "main.py" file).

    :param path: the path to the file in which the network is to be saved.
    :return: None
    """

    count = 1

    path = Path(__file__).parent.joinpath(path).with_suffix(".nn")

    with path.open(mode='wb') as dump_file:
        dump(file=dump_file, obj=neural_network)
