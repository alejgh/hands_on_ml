import numpy as np
import tensorflow as tf

from invalid_operation import InvalidOperationError


class TFLogisticRegressor():
    def __init__(self, batch_size=100, n_epochs=1000,
                 random_seed=42, learning_rate=0.01):
        self.X = tf.placeholder(tf.float64, None)
        self.y = tf.placeholder(tf.float64, (None, 1))
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.rnd_seed = random_seed
        self.learning_rate = learning_rate
        self.theta = None

    def fit(self, X, y):
        n = np.shape(X)[0]
        m = np.shape(X)[1]

        X = np.concatenate((np.ones((n, 1)), X), axis=1)
        y = np.reshape(y, (n, 1))

        theta = tf.Variable(tf.random_uniform([m + 1, 1], -1.0,
                            1.0, tf.float64, self.rnd_seed), name='theta')
        y_proba = tf.sigmoid(tf.matmul(X, theta))
        loss = tf.losses.log_loss(y, y_proba)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        training_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()

        n_batches = n // self.batch_size
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.n_epochs):
                for batch_idx in range(n_batches):
                    X_batch, y_batch = self.fetch_next_batch(X, y, batch_idx)
                    sess.run(training_op, feed_dict={self.X: np.asarray(X_batch),
                                                     self.y: np.asarray(y_batch)})
            self.theta = theta.eval()

    def fetch_next_batch(self, X, y, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        return X[start_idx:end_idx, :], y[start_idx:end_idx, :]

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return [0 if p < 0.5 else 1 for p in np.ravel(probabilities)]

    def predict_proba(self, X):
        if self.theta is None:
            raise InvalidOperationError('You must call fit before predict!')

        # add bias term
        n = np.shape(X)[0]
        X = np.concatenate((np.ones((n, 1)), X), axis=1)

        return 1 / (1 + np.exp(np.dot(-X, self.theta)))
