"""
Copyright (c) 2015-2016 Constantine Belev



Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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

    def error(self):
        if self.grad is not None:
            return self.grad.release().frobenius_norm()
        else:
            raise ValueError("gradient must exist.")

