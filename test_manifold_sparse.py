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

from matplotlib import pyplot as plt

from scipy.sparse import csr_matrix, lil_matrix

from riemannian_optimization.sparse.gd import gd_approximate, momentum_approximate
from riemannian_optimization.utils.test_utils import generate_sigma_set

np.set_printoptions(linewidth=450, suppress=True)

import cProfile

if __name__ == "__main__":
    shape = (10, 10)
    percent = 0.8
    sigma_set = generate_sigma_set(shape, percent)
    r = 2
    a_full = shape[1]*np.arange(shape[0])[:, None] + np.arange(shape[1])
    a_sparse = lil_matrix(shape)
    for (i, j) in zip(*sigma_set):
        a_sparse[i, j] = a_full[i, j]
    a_sparse = csr_matrix(a_sparse)
    x, it, err = momentum_approximate(a_sparse, sigma_set, r, maxiter=900)
    #cProfile.run('x, it, err = gd_approximate(a_sparse, sigma_set, r, maxiter=200)')
    #x, it, err = gd_approximate(a_sparse, sigma_set, r, maxiter=5)

    print('norm of x - a: {}'.format(np.linalg.norm(x.full_matrix() - a_full)))

    print('full matrix x:')
    print(x.full_matrix())

    print('-'*80)

    print('full matrix a: ')
    print(a_full)

    print('-'*80)

    print('delta x and a')
    print(x.full_matrix() - a_full)

    plt.plot(np.arange(len(err))[100:], err[100:])
    plt.show()
