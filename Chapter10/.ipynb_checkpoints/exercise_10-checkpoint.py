from tensorflow.examples.tutorials.mnist import input_data

from multilayer_perceptron import MultilayerPerceptron

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mlp = MultilayerPerceptron()
mlp.fit(mnist.train.images, mnist.train.labels)
