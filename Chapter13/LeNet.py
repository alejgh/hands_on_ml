import numpy as np
import tensorflow as tf
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import train_test_split

from functools import partial
from utils import fetch_next_batch


IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 1


class LeNet(BaseEstimator, ClassifierMixin):
    """Implementation of the LeNet CNN architecture.

    For more info see http://yann.lecun.com/exdb/lenet/
    """

    def __init__(self, activation=tf.nn.relu, batch_size=100, num_epochs=1000,
                 initializer=tf.contrib.layers.variance_scaling_initializer()):
        """Lenet CNN constructor.

        Args:
        activation: activation function that each layer in the network will use
        initializer: kernel initializer for the weights of each layer
        """
        self.activation = activation
        self.initializer = initializer
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def _cnn(self, input):
        conv_layer = partial(tf.layers.conv2d,
                             kernel_initializer=self.initializer,
                             padding="valid", activation=self.activation)
        pooling_layer = partial(tf.layers.average_pooling2d, pool_size=2,
                                strides=2, padding="valid")

        c1 = conv_layer(input, filters=6, kernel_size=5, strides=1, name="c1")
        s2 = pooling_layer(c1, name="s2")
        c3 = conv_layer(s2, filters=16, kernel_size=5, strides=1, name="c3")
        s4 = pooling_layer(c3, name="s4")
        c5 = conv_layer(s4, filters=120, kernel_size=5, strides=1, name="c5")
        c5_flat = tf.reshape(c5, shape=[-1, 120])
        return c5_flat

    def _build_graph(self, num_classes):
        _X = tf.placeholder(tf.float32, [None, IMG_HEIGHT * IMG_WIDTH])
        X_reshaped = tf.reshape(_X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS],
                                name="x_input_reshaped")
        _y = tf.placeholder(tf.int64, [None], name="y_input")

        with tf.name_scope("nn"):
            cnn_output = self._cnn(X_reshaped)
            f6 = tf.layers.dense(cnn_output, units=84,
                                 kernel_initializer=self.initializer,
                                 activation=self.activation, name="f6")
            output = tf.layers.dense(f6, units=num_classes, name="out")

        with tf.name_scope("loss"):
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                     labels=_y, logits=output), name="loss")

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss_op)

        with tf.name_scope('accuracy'):
            y_proba = tf.nn.softmax(output, name="y_proba")
            y_pred = tf.argmax(y_proba, axis=1)
            correct = tf.equal(y_pred, _y)
            accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32),
                                         name="accuracy")
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        self._X = _X
        self._y = _y
        self.y_pred = y_pred
        self.loss_op = loss_op
        self.train_op = train_op
        self.accuracy_op = accuracy_op
        self.saver = saver
        self.init = init

    def fit(self, X, y):
        """Fits the CNN to the given training data.

        Args:
        X: Array of images. Each image must be 32x32 and have just 1 channel
        y: Array of labels for each image
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
        num_classes = np.shape(np.unique(y))[0]

        graph = tf.Graph()
        with graph.as_default():
            self._build_graph(num_classes)

        self.session = tf.Session(graph=graph)
        self.session.run(self.init)

        best_loss = np.inf
        epochs_no_progress = 0
        MAX_EPOCHS_NO_PROGRESS = 20
        num_images = np.shape(X_train)[0]
        num_batches = num_images // self.batch_size
        with self.session.as_default() as sess:
            for epoch in range(self.num_epochs):
                for batch_idx in range(num_batches):
                    X_batch, y_batch = fetch_next_batch(X_train, y_train, batch_idx,
                                                        self.batch_size)
                    sess.run(self.train_op, feed_dict={self._X: X_batch,
                                                       self._y: y_batch})
                accuracy, loss = sess.run([self.accuracy_op, self.loss_op],
                                          feed_dict={self._X: X_val,
                                                     self._y: y_val})
                if loss < best_loss:
                    best_loss = loss
                    epochs_no_progress = 0
                    self.saver.save(self.session, 'LeNet/best_model.ckpt')
                else:
                    epochs_no_progress += 1
                    if epochs_no_progress == MAX_EPOCHS_NO_PROGRESS:
                        print('No progress after {} epochs. Stopping...'
                              .format(MAX_EPOCHS_NO_PROGRESS))
                        return
                print('{} - Loss: {:7f} - Best loss: {:.7f} - Accuracy: {:.3f}'
                      .format(epoch, loss, best_loss, accuracy))

    def predict(self, X):
        """Predicts the output class of each of the images passed as input

        Args:
        X: Array of n images. Each image must have 32x32 pixels and 1 channel.

        Returns:
        Array of n labels (1 per image) that represent the output class of each
        input image.
        """
        with self.session.as_default() as sess:
            return sess.run(self.y_pred, feed_dict={self._X: X})
