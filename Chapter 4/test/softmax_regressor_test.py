import unittest

import numpy as np

from src.softmax_regressor import CustomSoftmaxRegressor


class TestSoftmaxRegression(unittest.TestCase):

    def setUp(self):
        self.clf = CustomSoftmaxRegressor()

    def test_softmax_score(self):
        X = np.asarray(
            [[1, 2, 3],
             [5, 2, 1],
             [4, 2, 5],
             [2, 5, 1]]
        )
        theta = np.asarray(
            [[1, 1, 1],
             [1, 5, 6]]
        )
        result = np.asarray(
            [[6, 8, 11, 8],
             [29, 21, 4, 33]]
        )

        self.clf.weights = theta
        for i in range(np.shape(theta)[0]):
            self.assertEqual(result[i, :], self.clf._softmax_score(X, i))

    def test_compute_cost(self):
        pass

    def test_softmax_proba(self):
        pass


if __name__ == '__main__':
    unittest.main()
