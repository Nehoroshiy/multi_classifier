"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np
import scipy as sp

from scipy import sparse
from utils import generate_sigma_set
from scipy.sparse import csr_matrix, lil_matrix

from manopt import ManifoldElement
from manopt.approximator.manifold_functions import riemannian_grad_partial, vector_transport_base


def obvious_test(shape, r, niter=10):
    percent = 0.5
    sigma_set = generate_sigma_set(shape, percent)
    a_full = shape[1]*np.arange(shape[0])[:, None] + np.arange(shape[1])
    a_sparse = lil_matrix(shape)
    for (i, j) in zip(*sigma_set):
        a_sparse[i, j] = a_full[i, j]
    a_sparse = csr_matrix(a_sparse)
    x = ManifoldElement.rand(shape, r)
    grad = riemannian_grad_partial(x, a_sparse, sigma_set, manifold_elems=True)
    new_grad = vector_transport_base(x, x, grad)
    for elem, new_elem in zip(grad, new_grad):
        assert(np.allclose(elem.full_matrix(), new_elem.full_matrix()))


if __name__ == "__main__":
    args = ((100, 50), 10, 10)
    obvious_test(*args)
