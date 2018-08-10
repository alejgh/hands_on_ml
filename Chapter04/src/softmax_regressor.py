import sys

import numpy as np


class CustomSoftmaxRegressor():

    def __init__(self):
        self.weights = []
        self.Y = [[]]
        self.num_classes = -1

    def fit(self, X, y):
        m = np.shape(X)[1]

        self._compute_classes_matrix(y)
        self.weights = np.zeros((self.num_classes, m))

        delta = np.zeros((self.num_classes, m))
        error = sys.maxsize
        num_iter = 1
        while (error > 1e-5 and num_iter < 500):
            for i in range(self.num_classes):
                # vectorized implementation. Faster but harder to understand
                tmp = np.tile(self._softmax_proba(X, i) - self.Y[:, i],
                              (np.shape(X)[1], 1))
                delta[i, :] = np.sum(np.transpose(tmp) * X, axis=0) / m
                self.weights[i, :] -= delta[i, :]
            error = self._compute_cost(X, self.Y)
            num_iter += 1

    def predict(self, X):
        prob_matrix = np.zeros((np.shape(X)[0], self.num_classes))
        for i in range(self.num_classes):
            prob_matrix[:, i] = self._softmax_score(X, i)
        print(prob_matrix)
        return np.max(X, axis=1)

    def _softmax_proba(self, X, k):
        """
        Computes the probability that each sample in the
        data matrix X has of belonging to class k.
        """
        total_score = np.zeros(np.shape(X)[0])
        k_score = np.exp(self._softmax_score(X, k))
        for i in range(self.num_classes):
            total_score += np.exp(self._softmax_score(X, i))
        return k_score / total_score

    def _softmax_score(self, X, k):
        assert 0 <= k < self.num_classes
        return np.dot(self.weights[k], np.transpose(X))

    def _compute_classes_matrix(self, y):
        _, counts = np.unique(y, return_counts=True)
        m = np.shape(y)[0]
        self.num_classes = np.size(counts)
        self.Y = np.zeros((m, self.num_classes))
        for i in range(m):
            self.Y[i, :] = [1 if y[i] == k else 0 for k in range(self.num_classes)]

    def _compute_cost(self, X, Y):
        m = np.shape(X)[0]
        cost = 0
        for i in range(self.num_classes):
            cost += np.sum(Y[:, i] * np.log(self._softmax_proba(X, i)))
        cost = -cost / m
        return cost
