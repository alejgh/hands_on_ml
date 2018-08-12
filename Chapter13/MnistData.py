import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial

IMG_WIDTH = IMG_HEIGHT = 28
IMG_CHANNELS = 1


class MnistSet():
    def __init__(self, X, y):
        self.images = X
        self.labels = y


class MnistData():
    def __init__(self, one_hot):
        self.one_hot = one_hot
        mnist = self._download_data()
        self.train = MnistSet(mnist.train.images, mnist.train.labels)
        self.val = MnistSet(mnist.validation.images, mnist.validation.labels)
        self.test = MnistSet(mnist.test.images, mnist.test.labels)

    def _download_data(self):
        return input_data.read_data_sets('MNIST_data/', one_hot=self.one_hot)

    @staticmethod
    def get_default_data(one_hot=True):
        return MnistData(one_hot)

    @staticmethod
    def get_LeNet_data(one_hot=True):
        padding = ((2, 2), (2, 2))
        mnist = MnistData(one_hot)
        pad_fn = partial(MnistData._pad_mnist, padding=padding)
        mnist.train.images = pad_fn(mnist.train.images)
        mnist.val.images = pad_fn(mnist.val.images)
        mnist.test.images = pad_fn(mnist.test.images)
        return mnist

    @staticmethod
    def _pad_mnist(data, padding):
        data2d = [np.reshape(image, (IMG_WIDTH, IMG_HEIGHT)) for image in data]
        data2d_padded = [np.pad(image, pad_width=padding, mode='constant', constant_values=0)
                         for image in data2d]
        return [image.flatten() for image in data2d_padded]
