"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np
import scipy as sp
import timeit

from scipy import sparse
from manopt import ManifoldElement

from scipy.sparse import linalg, csr_matrix, csc_matrix, coo_matrix


def generate_sigma_set(shape, percent):
    return sp.sparse.random(*shape, density=percent).nonzero()


def generate_sorted_sigma_set(shape, percent):
    sigma_set = generate_sigma_set(shape, percent)
    sigma_set[0][:] = sigma_set[0][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[0].argsort(kind='mergesort')]
    sigma_set[0][:] = sigma_set[0][sigma_set[0].argsort(kind='mergesort')]
    return sigma_set


def evaluator(shape, density, r, iters=10):
    index_sets = [generate_sorted_sigma_set(shape, density) for _ in range(iters)]
    ur, vr = np.random.normal(size=(shape[0], r)), np.random.normal(size=(r, shape[1]))
    elem = ManifoldElement((ur, vr), r)

    timings = np.zeros(iters)
    for it in range(iters):
        timings[it] = timeit.Timer(lambda: elem.evaluate(index_sets[it])).timeit(number=1)
        print('next timing: %s' % timings[it])
    print(timings)
    print('min: %s, mean: %s, max: %s' % (timings.min(), timings.mean(), timings.max()))
    return None


if __name__ == "__main__":
    import cProfile
    shape = (5000, 5000)
    r = 10
    density = (3.0 * sum(shape) * r) / np.prod(shape)
    print(density)
    cProfile.run('evaluator(shape, density, r)')