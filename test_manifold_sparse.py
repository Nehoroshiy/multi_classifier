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
from scipy.sparse import csr_matrix, lil_matrix

from manopt.approximator import CGApproximator, ManifoldElement, cg_fuck
from manopt.approximator.cg import cg


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
if __name__ == "__main__":
    shape = (100, 100)
    r = 2
    opt_nnz = 10. * r * sum(shape)
    percent = opt_nnz / np.prod(shape)
    print('percent: {}'.format(percent))
    sigma_set = generate_sigma_set(shape, percent)
    sigma_set[0][:] = sigma_set[0][sigma_set[1].argsort()]
    sigma_set[1][:] = sigma_set[1][sigma_set[1].argsort()]
    sigma_set[1][:] = sigma_set[1][sigma_set[0].argsort()]
    sigma_set[0][:] = sigma_set[0][sigma_set[0].argsort()]

    a_full = 1. * (shape[1]*np.arange(shape[0])[:, None] + np.arange(shape[1]))
    a_sparse = lil_matrix(shape)
    for (i, j) in zip(*sigma_set):
        a_sparse[i, j] = a_full[i, j]
    a_sparse = csr_matrix(a_sparse)
    #test_batch_size = 20
    #maxiter = 200
    #its = np.array([old_cg(a_sparse, sigma_set, r, maxiter=maxiter)[1] for _ in range(test_batch_size)])
    #print(its)
    #print('fault rate: {}'.format(np.average(its == maxiter)))
    #cProfile.run('x, it, err = gd_approximate(a_sparse, sigma_set, r, maxiter=200)')
    #x, it, err = gd_approximate(a_sparse, sigma_set, r, maxiter=5)
    #x, it, err = cg(a_sparse, sigma_set, r, maxiter=600)
    print('a nnz: {}'.format(a_sparse.size))
    print('real a norm: {}'.format(np.linalg.norm(a_full)))
    u = np.fromfile('u_factor').reshape((-1, 1))
    s = np.fromfile('s_factor')
    v = np.fromfile('v_factor').reshape((1, -1))


    #x0 = ManifoldElement((u, s, v), r=1)
    """
    x0=None
    x = x0
    maxiter_ordinary = 10
    for rank in range(1, r):
        current_maxiter = 1
        x, it, err = cg_fuck(a_sparse, sigma_set, rank, x0=x, maxiter=int(maxiter_ordinary * current_maxiter), eps=1e-9)
        if it != int(maxiter_ordinary * current_maxiter):
            r = rank
            break
    print('rank is {}'.format(r))
    print('x sigma:')
    print(x.s)
    print('eps of x - a: {}'.format(np.linalg.norm(x.full_matrix() - a_full) / np.linalg.norm(a_full)))
    r = 2
    print('real rank is {}'.format(r))
    x = ManifoldElement(x, r)
    x, it, err = cg_fuck(a_sparse, sigma_set, r, x0=x, maxiter=100, eps=1e-14)
    print('rank is {}'.format(r))
    """
    approximator = CGApproximator()
    x, it, err = approximator.approximate(a=a_sparse, r=2, sigma_set=sigma_set, eps=1e-14)
    print('eps of x - a: {}'.format(np.linalg.norm(x.full_matrix() - a_full) / np.linalg.norm(a_full)))


    print('full matrix x:')
    print(x.full_matrix())

    print('-'*80)

    print('full matrix a: ')
    print(a_full)

    print('-'*80)

    print('delta x and a')
    print(x.full_matrix() - a_full)

    plt.plot(np.arange(len(err))[:], err[:])
    plt.show()