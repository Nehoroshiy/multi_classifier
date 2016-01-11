"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np


class AbstractApproximator():
    def __init__(self):
        self.grad = None
        self.sigma_set = None
        self.density = None
        self.norm_bound = None

    def approximate(self, a, r, sigma_set=None, x0=None, maxiter=900, eps=1e-9):
        raise NotImplementedError()

    def initialization(self, sigma_set=None):
        if sigma_set is None:
            self.sigma_set = self.target_matrix.nonzero()
        else:
            self.sigma_set = sigma_set
        self.sigma_set[0][:] = self.sigma_set[0][self.sigma_set[1].argsort(kind='mergesort')]
        self.sigma_set[1][:] = self.sigma_set[1][self.sigma_set[1].argsort(kind='mergesort')]
        self.sigma_set[1][:] = self.sigma_set[1][self.sigma_set[0].argsort(kind='mergesort')]
        self.sigma_set[0][:] = self.sigma_set[0][self.sigma_set[0].argsort(kind='mergesort')]

        self.density = 1.0 * len(self.sigma_set[0]) / np.prod(self.target_matrix.shape)
        part_norm = np.linalg.norm(np.array(self.target_matrix[self.sigma_set]))
        self.norm_bound = part_norm / np.sqrt(self.density)
        print('est. norm: %s' % self.norm_bound)
        return None

    def grad_projection(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def loss(self):
        if self.grad is not None:
            return self.grad.release().frobenius_norm()
        else:
            raise ValueError("gradient must exist.")

