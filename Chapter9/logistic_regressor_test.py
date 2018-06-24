import unittest
import numpy as np

from invalid_operation import InvalidOperationError
from logistic_regressor import TFLogisticRegressor


class TestTFLogisticRegressor(unittest.TestCase):
    def setUp(self):
        self.regressor = TFLogisticRegressor()
        self.X_train = [
            [2.5, 1.2, 0.03],
            [1.25, 12, 0.03],
            [1.35, 3.42, 12.23],
            [13, 52, 5.2]
        ]
        self.y_train = [0, 1, 1, 1]
        self.X_test = [[0.5, 0.1, -0.02]]

    def test_init(self):
        self.assertEqual(100, self.regressor.batch_size)
        self.assertEqual(1000, self.regressor.n_epochs)
        self.assertEqual(42, self.regressor.rnd_seed)
        custom_regressor = TFLogisticRegressor(batch_size=25, n_epochs=2)
        self.assertEqual(25, custom_regressor.batch_size)
        self.assertEqual(2, custom_regressor.n_epochs)
        self.assertEqual(42, custom_regressor.rnd_seed)

    def test_predict_fails_without_fit(self):
        with self.assertRaises(InvalidOperationError):
            self.regressor.predict(self.X_test)

    def test_fetch_batch(self):
        array_size = 5000
        x = np.random.randn(array_size, 50)
        y = np.random.randn(array_size, 1)
        batch_size = self.regressor.batch_size
        num_batches = array_size // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x_batch, y_batch = self.regressor.fetch_next_batch(x, y, i)
            x_expected, y_expected = x[start_idx:end_idx, :],\
                y[start_idx:end_idx, :]

            np.testing.assert_array_equal(x_expected, x_batch)
            np.testing.assert_array_equal(y_expected, y_batch)
            self.assertEqual(batch_size, np.shape(x_expected)[0])
            self.assertEqual(batch_size, np.shape(y_expected)[0])


if __name__ == '__main__':
    unittest.main()
