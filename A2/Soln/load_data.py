# adapted from
# https://github.com/fchollet/keras/blob/master/keras/datasets/cifar10.py

import cPickle
import os
import sys
import tarfile
from urllib2 import URLError

import numpy as np
from six.moves.urllib.request import urlretrieve


def get_file(file_name,
             origin,
             untar=False,
             extract=False,
             # archive_format='auto',
             cache_dir='data'):
    """
    Download a file from origin.

    :param file_name: the file name to download
    :type file_name: str
    :param origin: the origin for the file
    :type origin: str
    :param untar: true if untar the file
    :type untar: bool
    :param extract: true if extract the file
    :type extract: bool
    :param cache_dir: the path to the destination dir
    :type cache_dir: str
    :return: the saved file's path
    :rtype: str
    """
    data_dir = os.path.join(cache_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    untar_file_path = os.path.join(data_dir, file_name)
    if untar:
        file_path = untar_file_path + '.tar.gz'
    else:
        file_path = os.path.join(data_dir, file_name)

    print("Loading from " + file_path)
    if not os.path.exists(file_path):
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, file_path)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise Exception(error_msg.format(origin, e.errno, e.reason))

    if untar:
        if not os.path.exists(untar_file_path):
            print('Extracting file.')
            with tarfile.open(file_path) as archive:
                archive.extractall(data_dir)
        return untar_file_path

    if extract:
        # dunno what happened here this method is just gone
        # _extract_archive(file_path, data_dir, archive_format)
        pass

    return file_path


def _load_batch(file_path, label_key='labels'):
    """
    Internal utility for parsing CIFAR data.

    :param file_path: path the file to parse.
    :type file_path: str
    :param label_key: key for label data in the retrieve dictionary.
    :type label_key: str
    :return: A tuple `(data, labels)`.
    :rtype: tuple
    """
    f = open(file_path, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        # noinspection PyArgumentList
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_cifar10(transpose=False):
    """
    Loads CIFAR10 dataset.

    :param transpose: Get the transpose of the data sets
    :type transpose: bool
    :return: Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    :rtype: tuple
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin, untar=True)

    num_train_samples = 50000
    num_data_in_batch = 10000

    # training batches
    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        file_path = os.path.join(path, 'data_batch_' + str(i))
        data, labels = _load_batch(file_path)
        x_train[(i - 1) * num_data_in_batch: i * num_data_in_batch, :, :, :] = data
        y_train[(i - 1) * num_data_in_batch: i * num_data_in_batch] = labels

    # test batch
    file_path = os.path.join(path, 'test_batch')
    x_test, y_test = _load_batch(file_path)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if transpose:
        x_train = x_train.transpose((0, 2, 3, 1))
        x_test = x_test.transpose((0, 2, 3, 1))
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    m = load_cifar10()
