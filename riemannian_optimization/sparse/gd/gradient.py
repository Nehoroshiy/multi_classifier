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
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, linalg

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

    Projection has the form
    $Proj(Z) = UU^*Z + ZVV^* + UU^*ZVV^*$

    Parameters
    ----------
    x : ManifoldElement, shape (M, N)
        Rank-r manifold element in which we compute gradient
    a : sparse matrix, shape (M, N)
        Matrix that we need to approximate -- it has nonzero entries only
        on sigma_set
    sigma_set : array_like
        set of indices in which matrix a can be evaluated
    grad : sparse matrix, shape (M, N), optional
        gradient given for being projected

    Returns
    -------
    out : ManifoldElement
        Projection of an Euclidean gradient onto the Tangent space at x
    """
    grad = ManifoldElement(euclid_grad(x, a, sigma_set)) if grad is None else grad
    left_projected = grad.rdot(x.u.T).rdot(x.u)
    right_projected = grad.dot(x.v.T).dot(x.v)
    return left_projected + right_projected + left_projected.dot(x.v.T).dot(x.v)


def retract(x, r):
    """
    Returns given tangent space element back to rank-r manifold.

    In current version, retraction is proceeded by truncated SVD decomposition

    Parameters
    ----------
    x : ManifoldElement, shape (M, N)
        element to perform retraction
    r : int
        rank of manifold

    Returns
    -------
    out : ManifoldElement, shape (M, N)
        element of rank-r manifold, retraction of x onto it
    """
    return ManifoldElement(x, r)


def gd_approximate(a, sigma_set, r, maxiter=900, eps=1e-9):
    """
    Approximation of sparse matrix a with gradient descend method.
    Matrix a is known only at sigma_set indices, at other indices it equals zero.
    Function use Riemannian Gradient Descend with line search to find approximation.

    Parameters
    ----------
    a : sp.sparse.spmatrix, shape (M, N)
        Matrix being approximated
    sigma_set : np.array_like
        set of i-indices and j-indices
    r : int
        rank constraint
    maxiter : int, optional
        maximum number of iteration
    eps : float, optional
        tolerance for gradient norm checking

    Returns
    -------
    out : ManifoldElement, shape (M, N)
        Approximation found by algorithm
    iter: int
        Number of iterations spent by algorithms to reach approximation.
        If equals maxiter, then algorithm not converged
    err: list
        List of grad norms at each iteration
    """
    def cost_func(param):
        temp = x + param * projection
        return 0.5 * sp.sparse.linalg.norm(temp.evaluate(sigma_set) - a) ** 2

    x = ManifoldElement.rand(a.shape, r)
    err = []
    for it in range(maxiter):
        grad = euclid_grad(x, a, sigma_set)
        err.append(sp.sparse.linalg.norm(grad))
        if err[-1] < eps:
            print('Small grad norm {} is reached at iteration {}'.format(err[-1], it))
            return x, it, err
        projection = ManifoldElement(-riemannian_grad(x, a, sigma_set, grad=grad))
        alpha = minimize_scalar(fun=cost_func, bounds=(0., 10.), method='bounded')['x']
        x = retract(x + alpha * projection, r)
    print('Error {} is reached at iteration {}. Cannot converge'.format(err[-1], maxiter))
    return x, maxiter, err
