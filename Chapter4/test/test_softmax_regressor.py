import unittest

import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
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

        # theta * X'
        result = np.asarray(
            [[6,   8, 11,  8],
             [29, 21, 44, 33]]
        )

        self.clf.weights = theta  # simulate training
        self.clf.num_classes = np.shape(theta)[0]
        for i in range(self.clf.num_classes):
            assert_array_equal(result[i, :], self.clf._softmax_score(X, i))

    def test_compute_cost(self):
        X = np.asarray(
            [[1.4, 1.2, 4.0, 2.4],
             [6.1, 4.1, 2.3, 3.3],
             [4.5, 4.2, 1.24, 2.4]]
        )
        theta = np.asarray(
            [[1, 1, 4, 5],
             [2, 4, 5, 5]]
        )
        # 2 classes
        Y = np.asarray(
            [[1, 0],
             [0, 1],
             [0, 1]]
        )

        cost = 3.000041
        self.clf.weights = theta  # simulate training
        self.clf.num_classes = np.shape(theta)[0]
        assert_array_almost_equal(cost, self.clf._compute_cost(X, Y))

    def test_softmax_proba(self):
        X = np.asarray(
            [[6, 1, 5],
             [1, 5, 2],
             [2, 1, 5],
             [6, 7, 4]]
        )
        theta = np.asarray(
            [[2, 5, 6],
             [1, 2, 5],
             [3, 1, 3]]
        )
        proba = np.asarray(
            [[0.999, 8.315e-07, 2.26e-06],
             [0.999, 1.523e-08, 1.38e-11],
             [0.999, 4.539e-05, 4.13e-08],
             [1,     3.442e-14, 1.71e-15]]
        )

        self.clf.theta = theta
        assert_array_almost_equal(proba, self.clf.predict_proba(X), decimal=3)


if __name__ == '__main__':
    unittest.main()
