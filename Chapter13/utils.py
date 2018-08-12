import tensorflow as tf

he_init = tf.contrib.layers.variance_scaling_initializer()


def fetch_next_batch(X, y, batch_idx, batch_size):
    start = batch_idx * batch_size
    end = (batch_idx + 1) * batch_size
    batch_x = X[start:end]
    batch_y = y[start:end]
    return batch_x, batch_y
