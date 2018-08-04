from tensorflow.examples.tutorials.mnist import input_data

from multilayer_perceptron import MultilayerPerceptron

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_inputs = len(mnist.train.images)
n_classes = len(mnist.train.labels[0])
batch_size = n_inputs // 10

print('Num Inputs: {}'.format(n_inputs))
print('Num classes: {}'.format(n_classes))
print('Batch size: {}'.format(batch_size))

mlp = MultilayerPerceptron(n_inputs, n_classes, batch_size)
mlp.add_layer(50)
mlp.add_layer(20)
mlp.add_layer(10)
mlp.fit(mnist.train.images, mnist.train.labels)
pred = mlp.predict(mnist.test.images)
