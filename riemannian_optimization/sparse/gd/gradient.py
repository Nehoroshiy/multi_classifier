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

from scipy import sparse, linalg, optimize

from scipy.linalg import rq
from numpy.linalg import svd, qr, norm
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

from riemannian_optimization.utils.approx_utils import csvd
from riemannian_optimization.lowrank_matrix import ManifoldElement


def euclid_grad(x, a, sigma_set):
    """
    Euclidean gradient.

    Compute euclidean gradient of function $\dfrac{1}{2}| P_{\Sigma}(X - A)|_F^2$,
    equals to P_{\Sigma}(X - A).

    Parameters
    ----------
    x : ManifoldElement, shape (M, N)
        Rank-r manifold element in which we compute gradient
    a : sparse matrix, shape (M, N)
        Matrix that we need to approximate -- it has nonzero entries only
        on sigma_set
    sigma_set : array_like
        set of indices in which matrix a can be evaluated

    Returns
    -------
    grad: sparse matrix, shape (M, N)
        Gradient of our functional at x
    """
    if x.shape != a.shape:
        raise ValueError("shapes of x and a must be equal")
    return x.evaluate(sigma_set) - a


def riemannian_grad(x, a, sigma_set, grad=None):
    """
    Riemannian gradient

    Compute projection of Euclidean gradient of function
    $\dfrac{1}{2}| P_{\Sigma}(X - A)|_F^2$ at tangent space to manifold at x.

    Parameters
    ----------
    x : ManifoldElement, shape (M, N)
        Rank-r manifold element in which we compute gradient
    a : sparse matrix, shape (M, N)
        Matrix that we need to approximate -- it has nonzero entries only
        on sigma_set
    sigma_set : array_like
        set of indices in which matrix a can be evaluated
    grad

    Returns
    -------

    """