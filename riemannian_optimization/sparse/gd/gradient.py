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
from scipy.sparse import linalg
from scipy.optimize import minimize_scalar

from ..utils.projections import riemannian_grad_full
from ..utils.loss_functions import delta_on_sigma_set
from ..utils.retractions import svd_retraction as retraction

from riemannian_optimization.lowrank_matrix import ManifoldElement


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

    density = 1.0 * len(sigma_set[0]) / np.prod(a.shape)
    x = ManifoldElement.rand(a.shape, r,
                             desired_norm=np.linalg.norm(a[sigma_set]) / np.sqrt(density))
    err = []
    for it in range(maxiter):
        grad = delta_on_sigma_set(x, a, sigma_set)
        err.append(sp.sparse.linalg.norm(grad))
        if err[-1] < eps:
            print('Small grad norm {} is reached at iteration {}'.format(err[-1], it))
            return x, it, err
        projection = riemannian_grad_full(x, a, sigma_set, grad=-grad)
        alpha = minimize_scalar(fun=cost_func, bounds=(0., 5.), method='bounded')['x']
        print('iter:{}, alpha: {}, error: {}'.format(it, alpha, err[-1]))
        x = retraction(x + alpha * projection, r)
    print('Error {} is reached at iteration {}. Cannot converge'.format(err[-1], maxiter))
    return x, maxiter, err
