import numpy as np
import scipy as sp

from scipy import linalg
from matplotlib import pyplot as plt
import seaborn as sns
from riemannian_optimization import ManifoldElement

from numpy.linalg import norm, matrix_rank as mrank

from riemannian_optimization.gd.gradient import approximate

np.set_printoptions(linewidth=450, suppress=True)

def all_indices(m, n):
    mg = np.meshgrid(np.arange(n), np.arange(m))
    mg = map(lambda x: x.ravel(), mg[::-1])
    return mg

def part(indices, percent):
    perm = np.random.permutation(indices[0].size)
    return map(lambda x: x[perm][:int(indices[0].size * percent)], indices)

if __name__ == "__main__":
    """
    m, n = 10, 5
    M = 50
    percent = .8
    a = np.random.random((m, n))
    """
    m, n = (10, 10)
    M = 50
    percent = 0.8
    a = 10*np.arange(m)[:, None] + np.arange(n)
    #sigma_set = (np.random.choice(m, M, replace=True), np.random.choice(n, M, replace=True))

    sigma_set = part(all_indices(m, n), percent)
    print(sigma_set[0].size)
    r = 2
    it, x, err = approximate(a, sigma_set, r)
    print('norm of x - a: {}'.format(norm(x.full_matrix() - a)))

    print('full matrix x:')
    print(x.full_matrix())

    print('-'*80)

    print('full matrix a: ')
    print(a)

    print('-'*80)

    print('delta x and a')
    print(x.full_matrix() - a)

    plt.plot(np.arange(len(err))[100:], err[100:])
    plt.show()
"""
if __name__ == "__main__":
    m = 100
    n = 500
    #
    x = np.random.random((m, n))
    y = np.random.random((m, n))
    #
    mx = ManifoldElement(x, r=4)
    my = ManifoldElement(y, r=4)
    #
    x_full = mx.full_matrix()
    y_full = my.full_matrix()
    z_full_expected = x_full + y_full
    #
    mz = mx + my
    print(norm(mz.full_matrix() - z_full_expected))
"""