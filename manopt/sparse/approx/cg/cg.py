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

from conjugate_direction import conjugate_direction
from scipy.sparse import linalg, csr_matrix, csc_matrix
from scipy.optimize import minimize_scalar, line_search
from manopt.lowrank_matrix import ManifoldElement
from manopt.sparse.utils.retractions import svd_retraction
from manopt.sparse.utils.loss_functions import delta_on_sigma_set
from manopt.sparse.utils.projections import TangentVector
from manopt.sparse.utils.projections import riemannian_grad_partial, riemannian_grad_full


def closed_form_initial_guess(vec, delta, sigma_set):
    n_mat = csc_matrix(vec.release().evaluate(sigma_set)).T
    trace_first = n_mat.multiply(delta.T).sum()
    trace_second = n_mat.multiply(n_mat).sum()
    return np.abs(trace_first / trace_second)


def armijo_backtracking(func, x, alpha, direction, conj_direction, maxiter=20):
    """
    Returns step and next point, minimizing given functional

    Parameters
    ----------
    func : function
        function to minimize
    x : ManifoldElement
        initial point
    alpha : float
        estimated line search parameter
    direction : TangentVector
        direction to move from initial point
    conj_direction : TangentVector
        conjugated direction

    Returns
    -------
    x_new :
        next point (x + step * direction)
    step : float
        optimal step
    """
    scale = -0.0001 * alpha
    for i in range(maxiter):
        x_new = svd_retraction(x + (0.5**i * scale) * conj_direction.release(), x.r)
        bound = (0.5**i * scale) * direction.release().scalar_product(conj_direction.release())
        if cost_raw(x) - cost_raw(x_new) >= bound:
            return x_new, 0.5**i * scale
    return x_new, 0.5**maxiter * scale


def cost_raw(a, elem, sigma_set):
    """
    Compute function 0.5 *|| a[sigma] - elem[sigma] ||_F^2

    Parameters
    ----------
    a : np.ndarray or sp.sparse.spmatrix
        matrix to approximate
    elem : ManifoldElement
        approximation
    sigma_set : tuple of np.ndarrays
        index set of x indices and y indices

    Returns
    -------
    out: float
        cost function
    """
    return 0.5 * sp.sparse.linalg.norm(elem.evaluate(sigma_set) - a) ** 2


def cg(a, sigma_set, r, x0=None, maxiter=900, eps=1e-9):
    density = 1.0 * len(sigma_set[0]) / np.prod(a.shape)
    norm_bound = np.linalg.norm(np.array(a[sigma_set])) / np.sqrt(density)

    if x0 is None:
        x0 = ManifoldElement.rand(a.shape, r, norm=norm_bound)
    x_, x = x0, x0

    grad = -TangentVector(x0, riemannian_grad_partial(x0, a, sigma_set, manifold_elems=True))
    conj_, conj = TangentVector.zero(x), TangentVector.zero(x)
    error_history = []

    for it in range(maxiter):
        # euclidean gradient
        delta = delta_on_sigma_set(x, a, sigma_set)
        # tangent space projection
        riemannian_grad = riemannian_grad_partial(x, a, sigma_set, grad=delta, manifold_elems=True)
        grad_, grad = grad, -TangentVector(x, riemannian_grad)

        # check gradient error
        error_history.append(grad.release().frobenius_norm())
        if error_history[-1] < eps * norm_bound:
            return x, it, error_history
        # compute conjugate direction
        conj_, conj = conj, conjugate_direction(x_, grad_, conj_, x, grad)

        # estimate step size and compute next approximation
        # using armijo backtracking
        alpha = closed_form_initial_guess(conj, delta, sigma_set)
        x_, x = x, armijo_backtracking(cost_raw, x, alpha, grad, conj)[0]
    # iteration fails
    return x, maxiter, error_history


def cg(a, sigma_set, r, x0=None, maxiter=900, eps=1e-9):

    # estimate norm of matrix
    density = 1.0 * len(sigma_set[0]) / np.prod(a.shape)
    norm_bound = np.linalg.norm(np.array(a[sigma_set])) / np.sqrt(density)
    print('norm of %s part: %s, est. norm: %s' %
          (density, norm_bound * np.sqrt(density), norm_bound))

    # initial guess
    if x0 is None:
        x0 = ManifoldElement.rand(a.shape, r, norm=norm_bound)
    x_, x = x0, x0

    # initialization
    grad = -TangentVector(x0, riemannian_grad_partial(x0, a, sigma_set, manifold_elems=True))
    conj_, conj = TangentVector.zero(x), TangentVector.zero(x)
    conj_mat = conj.release()
    error_history = []

    for it in range(maxiter):
        delta = delta_on_sigma_set(x, a, sigma_set)
        riemannian_grad = riemannian_grad_partial(x, a, sigma_set, grad=delta, manifold_elems=True)
        grad_, grad = grad, -TangentVector(x, riemannian_grad)

        error_history.append(grad.release().frobenius_norm())
        if error_history[-1] < eps * norm_bound:
            print('Small grad norm {} is reached at iteration {}'.format(error_history[-1], it))
            return x, it, error_history
        conj_, conj = conj, conjugate_direction(x_, grad_, conj_, x, grad)

        alpha = closed_form_initial_guess(conj, delta, sigma_set)
        x_new, alpha = armijo_backtracking(cost_raw, x, alpha, grad, conj)
        print('iter:{}, alpha: {}, error: {}'.format(it, alpha, error_history[-1]))
        x_, x = x, x_new
    print('Error {} is reached at iteration {}. Cannot converge'.format(error_history[-1], maxiter))
    return x, maxiter, error_history