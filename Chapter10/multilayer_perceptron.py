import tensorflow as tf
import numpy as np

import os
from datetime import datetime


class MultilayerPerceptron():
    def __init__(self, n_inputs, n_classes, batch_size,
                 learning_rate=1e-4, num_epochs=100):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.X = tf.placeholder(tf.float32, [None, 784], "x_input")
        self.y = tf.placeholder(tf.int32, [None, n_classes], "y_input")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.session = tf.Session()
        self.layers = []

        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        log_dir = os.path.join("tf_logs", "run-{}".format(now))
        self.writer = tf.summary.FileWriter(log_dir)

    def add_layer(self, num_neurons, activation=None, name="layer"):
        with tf.name_scope(name):
            if (len(self.layers) == 0):
                n_inputs = 784
                stddev = 2 / np.sqrt(n_inputs)
                weights = tf.Variable(tf.random_normal([784, num_neurons],
                                      stddev=stddev), name="weights")
                biases = tf.Variable(tf.zeros([num_neurons]),
                                     name="bias")
                new_layer = tf.matmul(self.X, weights) + biases
            else:
                prev_layer = self.layers[-1]
                n_inputs = int(prev_layer.shape[1])
                stddev = 2 / np.sqrt(n_inputs)
                weights = tf.Variable(tf.random_normal([n_inputs, num_neurons],
                                                       stddev=stddev))
                biases = tf.Variable(tf.zeros([num_neurons]))
                new_layer = tf.matmul(prev_layer, weights) + biases

            if activation is not None:
                new_layer = activation(new_layer)
            self.layers.append(new_layer)

    def fit(self, X, y):
        with tf.name_scope('xent'):
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.layers[-1], labels=self.y))

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            traininig_op = optimizer.minimize(loss_op)

        with tf.name_scope('eval'):
            correct = tf.equal(tf.argmax(self.layers[-1], axis=1), tf.argmax(self.y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('accuracy', accuracy)
        summaries = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        num_samples = self.n_inputs
        self.writer.add_graph(self.session.graph)
        self.session.run(init)
        for epoch in range(self.num_epochs):
            avg_cost = 0
            avg_acc = 0
            num_batches = num_samples // self.batch_size
            for batch_idx in range(num_batches):
                batch_x, batch_y = self.fetch_batch(batch_idx, X, y)
                _, cost, acc = self.session.run([traininig_op, loss_op, accuracy],
                                                feed_dict={self.X: batch_x,
                                                           self.y: batch_y})
                avg_cost += cost / num_batches
                avg_acc += acc / num_batches
                if batch_idx % 3 == 0:
                    step = epoch * num_batches + batch_idx
                    s = self.session.run(summaries, feed_dict={self.X: batch_x,
                                                               self.y: batch_y})
                    self.writer.add_summary(s, step)
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost),
                  "acc={:.9f}".format(avg_acc))

    def predict(self, X):
        softmax = tf.nn.softmax(self.layers[-1])
        prediction = tf.argmax(softmax, 1)
        return prediction.eval(session=self.session, feed_dict={self.X: X})

    def fetch_batch(self, batch_idx, X, y):
        start = batch_idx * self.batch_size
        end = (batch_idx + 1) * self.batch_size
        batch_x = X[start:end]
        batch_y = y[start:end]
        return batch_x, batch_y
