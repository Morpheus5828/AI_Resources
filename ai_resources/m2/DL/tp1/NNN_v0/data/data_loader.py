"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

import pickle as cPickle
import gzip
from urllib.request import urlretrieve
from utils import one_hot_encoding
import numpy as np

"""Load the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

"""
    
def download_data():
    """Download dataset in pkl.gz format"""
    
    try:
        f = gzip.open('data/mnist.pkl.gz', 'rb')
    except:
        print("downloading MNIST")
        urlretrieve("https://pageperso.lis-lab.fr/stephane.ayache/mnist.pkl.gz", "data/mnist.pkl.gz")
        f = gzip.open('data/mnist.pkl.gz', 'rb')
    
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data


def load_data(): # subsample to 10000 training data
    """Return a tuple containing ``(training_data, validation_data,
    test_data)`` more convenient that from ``load_data`` format.
    """
    
    tr_d, va_d, te_d = download_data()
    training_inputs = [np.reshape(x, (784,)) for x in tr_d[0]]
    training_data_zip = zip(training_inputs, tr_d[1])
    validation_inputs = [np.reshape(x, (784,)) for x in va_d[0]]
    validation_data_zip = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784,)) for x in te_d[0]]
    test_data_zip = zip(test_inputs, te_d[1])
    
    # format better training set
    training_data = np.array(list(training_data_zip), dtype="object")
    training_examples = np.apply_along_axis(lambda x: x[0], axis=1, arr=training_data)
    training_labels = np.apply_along_axis(lambda x: x[1], axis=1, arr=training_data)
    training_labels = one_hot_encoding(training_labels) #convert to one_hot

    # subsample training set
    idx = np.arange(0, len(training_labels), 1)
    np.random.shuffle(idx)
    idx = idx[:10000]
    training_examples = training_examples[idx]
    training_labels = training_labels[idx]
    print(training_examples.shape)

    # format better val set
    validation_data = np.array(list(validation_data_zip), dtype="object")
    validation_examples = np.apply_along_axis(lambda x: x[0], axis=1, arr=validation_data)
    validation_labels = np.apply_along_axis(lambda x: x[1], axis=1, arr=validation_data)
    validation_labels = one_hot_encoding(validation_labels) #convert to one_hot

    # format better test set
    test_data = np.array(list(test_data_zip), dtype="object")
    test_examples = np.apply_along_axis(lambda x: x[0], axis=1, arr=test_data)
    test_labels = np.apply_along_axis(lambda x: x[1], axis=1, arr=test_data)
    test_labels = one_hot_encoding(test_labels) #convert to one_hot
    
    return training_examples, training_labels, validation_examples, validation_labels, test_examples, test_labels

