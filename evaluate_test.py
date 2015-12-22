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