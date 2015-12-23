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
import scipy as sp
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

from manopt import BlackBox
from manopt.approximator import CGApproximator, GDApproximator, ManifoldElement, MGDApproximator


def generate_sigma_set(shape, percent):
    return sp.sparse.random(*shape, density=percent).nonzero()

np.set_printoptions(linewidth=450, suppress=True)

# TODO rewrite all this shitty code.
# 1. Move each variation of your gradient methods in separate files.
# 2. Write test that will compare behavior of your algorithms.
# 3. Create some tests (low rank matrices with ranks [2, 3, 4, 5, ...])
# 4. Try to test on Hilbert matrix
# 5. Try to make experimenting easier (unified way to make low-rank matrices,
#       unified way to perform and compare algorithms and they quality)


def gen_sum_func(shape, power=1):
    def sum_func(indices):
        return (indices[:, 0] + (1. / shape[1]) * indices[:, 1])**power
    return sum_func


def approx_test_ranks(shape, ranks=None, maxiter=200):
    approximator = CGApproximator()
    ranks = np.arange(1, 5) if ranks is None else np.asarray(ranks, dtype=int)
    boxes = [BlackBox(gen_sum_func(shape, power=r-1), shape) for r in ranks]
    results = []
    for r, box in zip(ranks, boxes):
        opt_nnz = 10. * r * sum(shape)
        percent = opt_nnz / np.prod(shape)
        print('percent: {}'.format(percent))
        sigma_set = generate_sigma_set(box.shape, percent)
        data = box[sigma_set]
        a_sparse = csr_matrix(coo_matrix((data, sigma_set), shape=shape))

        results.append(approximator.approximate(a_sparse, r, sigma_set, maxiter=maxiter, eps=1e-10))
    for i, (x, it, err) in enumerate(results):
        box = boxes[i]
        print('eps of x - a: {} at r={}'.format(np.linalg.norm(x.full_matrix() - box[:]) / np.linalg.norm(box[:]), i+1))
        err = np.pad(np.array(err) / np.linalg.norm(box[:]), (0, maxiter - len(err)), mode='constant')
        if i >= 1:
            plt.plot(np.arange(maxiter), err)
    plt.legend([r'r=%s' % r for r in ranks[1:]])
    plt.show()
    return None


def approx_test_random(approximator, shape, ranks=None, maxiter=200):
    ranks = np.arange(1, 5, 1)*1 if ranks is None else np.asarray(ranks, dtype=int)

    x = np.random.random(shape)
    u_lowrank = np.random.normal(size=(shape[0], max(ranks)))
    v_lowrank = np.random.normal(size=(max(ranks), shape[1]))

    results = []
    ys = []
    for r in ranks:
        y = u_lowrank[:, :r].dot(v_lowrank[:r, :])
        ys.append(y)
        opt_nnz = 5. * r * sum(shape)
        percent = opt_nnz / np.prod(shape)
        print('percent: {}'.format(percent))
        sigma_set = generate_sigma_set(y.shape, percent)
        data = y[sigma_set]
        a_sparse = csr_matrix(coo_matrix((data, sigma_set), shape=shape))

        results.append(approximator.approximate(a_sparse, r, sigma_set, maxiter=maxiter, eps=1e-10))

    for i, (x, it, err) in enumerate(results):
        box = ys[i]
        print('eps of x - a: {} at r={}'.format(np.linalg.norm(x.full_matrix() - box[:]) / np.linalg.norm(box[:]), i+1))
        err = np.pad(np.array(err) / np.linalg.norm(box[:]), (0, maxiter - len(err)), mode='constant')
        plt.semilogy(np.arange(maxiter), err)
    plt.legend([r'r=%s' % r for r in ranks[:]])
    plt.show()

    return None


if __name__ == "__main__":
    import cProfile
    #shapes = [(n, n) for n in 32*np.arange(1, 4)]
    ranks = np.arange(1, 5, 1) * 5
    shape = (400, 400)
    #approx_test_ranks(shape)
    cProfile.run('approx_test_random(CGApproximator(), shape, ranks[:])')
    #approx_test_random(GDApproximator(), shape, ranks[1:2])
