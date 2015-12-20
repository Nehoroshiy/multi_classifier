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
from basic_vector_transport import vector_transport_base

from scipy.sparse import csr_matrix, lil_matrix

from manopt import ManifoldElement
from manopt.sparse.utils.projections import riemannian_grad_partial
from manopt.utils.test_utils import generate_sigma_set


def obvious_test(shape, r, niter=10):
    percent = 0.5
    sigma_set = generate_sigma_set(shape, percent)
    a_full = shape[1]*np.arange(shape[0])[:, None] + np.arange(shape[1])
    a_sparse = lil_matrix(shape)
    for (i, j) in zip(*sigma_set):
        a_sparse[i, j] = a_full[i, j]
    a_sparse = csr_matrix(a_sparse)
    x = ManifoldElement.rand(shape, r)
    grad = riemannian_grad_partial(x, a_sparse, sigma_set, as_manifold_elements=True)
    new_grad = vector_transport_base(x, x, grad)
    for elem, new_elem in zip(grad, new_grad):
        assert(np.allclose(elem.full_matrix(), new_elem.full_matrix()))


if __name__ == "__main__":
    args = ((100, 50), 10, 10)
    obvious_test(*args)
