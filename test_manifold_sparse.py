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

from scipy import sparse
from matplotlib import pyplot as plt

from scipy.sparse import linalg, csr_matrix, lil_matrix

from riemannian_optimization.sparse.gd import gd_approximate

np.set_printoptions(linewidth=450, suppress=True)


def all_indices(m, n):
    mg = np.meshgrid(np.arange(n), np.arange(m))
    mg = list(map(lambda x: x.ravel(), mg[::-1]))
    return mg


def part(indices, percent):
    perm = np.random.permutation(indices[0].size)
    return list(map(lambda x: x[perm][:int(indices[0].size * percent)], indices))


if __name__ == "__main__":
    shape = (10, 10)
    percent = 0.9
    sigma_set = part(all_indices(*shape), percent)
    r = 2
    a_full = 10*np.arange(shape[0])[:, None] + np.arange(shape[1])
    a_sparse = lil_matrix(shape)
    for (i, j) in zip(*sigma_set):
        a_sparse[i, j] = a_full[i, j]
    a_sparse = csr_matrix(a_sparse)
    x, it, err = gd_approximate(a_sparse, sigma_set, r)
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
