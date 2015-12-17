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

from scipy.sparse import linalg, csr_matrix, csc_matrix
from scipy.optimize import minimize_scalar, line_search
from riemannian_optimization.lowrank_matrix import ManifoldElement
from riemannian_optimization.sparse.utils.retractions import svd_retraction
from riemannian_optimization.sparse.utils.loss_functions import delta_on_sigma_set
from riemannian_optimization.sparse.utils.projections import TangentVector, riemannian_grad_partial


def conjugate_direction(x_prev, grad_prev, dir_prev, x, grad):
    grad_prev_trans = grad_prev.transport(x)
    dir_prev_trans = dir_prev.transport(x)

    delta = grad - grad_prev_trans
    delta_released = delta.release()
    grad_prev_released = grad_prev_trans.release()
    grad_released = grad.release()
    beta = max(0, delta_released.scalar_product(grad_released) / grad_prev_released.frobenius_norm()**2)
    dir_ = -grad + beta * dir_prev_trans
    dir_released = dir_.release()

    angle = grad_released.scalar_product(dir_released) / \
            np.sqrt(dir_released.frobenius_norm()**2 * grad_released.frobenius_norm()**2)
    if angle <= 0.1:
        dir_ = grad
    return dir_


def closed_form_initial_guess(vec, delta, sigma_set):
    n_mat = csc_matrix(vec.release().evaluate(sigma_set)).T
    trace_first = n_mat.multiply(delta.T).sum()
    trace_second = n_mat.multiply(n_mat.T).sum()
    return trace_first / trace_second


def cg(a, sigma_set, r, maxiter=900, eps=1e-9):
    def cost_function(param):
        temp = x + param * conj_released
        return 0.5 * sp.sparse.linalg.norm(temp.evaluate(sigma_set) - a) ** 2

    def cost_raw(elem):
        return 0.5 * sp.sparse.linalg.norm(elem.evaluate(sigma_set) - a) ** 2

    x = ManifoldElement.rand(a.shape, r)
    x_prev = x
    error_history = []
    conj, conj_prev = TangentVector.zero(x), TangentVector.zero(x)
    conj_released, conj_prev_released = conj.release(), conj_prev.release()
    grad = -TangentVector(x, riemannian_grad_partial(x, a, sigma_set,
                                                     as_manifold_elements=True))
    grad_released = grad.release()
    for it in range(maxiter):
        delta = delta_on_sigma_set(x, a, sigma_set)
        grad_prev, grad = \
            grad, -TangentVector(x, riemannian_grad_partial(x, a, sigma_set, grad=delta,
                                                            as_manifold_elements=True))
        grad_prev_released, grad_released = grad_released, grad.release()
        error_history.append(grad_released.frobenius_norm())
        if error_history[-1] < eps:
            print('Small grad norm {} is reached at iteration {}'.format(error_history[-1], it))
            return x, it, error_history
        conj_prev, conj = conj, conjugate_direction(x_prev, grad_prev, conj_prev, x, grad)
        conj_prev_released, conj_released = conj_released, conj.release()
        alpha = minimize_scalar(fun=cost_function, bounds=(0., 10.), method='bounded')['x']
        m = 0
        for i in range(20):
            x_new = svd_retraction(x + (0.5**m * alpha) * conj_released, r)
            if cost_raw(x) - cost_raw(x_new) >= \
                    (-0.0001 * 0.5**m * alpha) * grad_released.scalar_product(conj_released):
                m = i
                break
        print('iter:{}, alpha: {}, m: {} error: {}'.format(it, alpha, m, error_history[-1]))
        x_prev, x = x, x_new
    print('Error {} is reached at iteration {}. Cannot converge'.format(error_history[-1], maxiter))
    return x, maxiter, error_history


def old_cg(a, sigma_set, r, maxiter=900, eps=1e-9):
    def cost_function(param):
        temp = x + param * conj_dir.release()
        return 0.5 * sp.sparse.linalg.norm(temp.evaluate(sigma_set) - a) ** 2

    def cost_raw(x_val):
        return 0.5 * sp.sparse.linalg.norm(x_val.evaluate(sigma_set) - a) ** 2


    x = ManifoldElement.rand(a.shape, r)
    x_prev = x
    error_history = []
    conj_dir, conj_dir_prev = TangentVector.zero(x), TangentVector.zero(x)
    grad = -TangentVector(x, riemannian_grad_partial(x, a, sigma_set, as_manifold_elements=True))
    for it in range(maxiter):
        delta = delta_on_sigma_set(x, a, sigma_set)
        grad_prev, grad = grad, -TangentVector(x, riemannian_grad_partial(x, a, sigma_set, grad=delta, as_manifold_elements=True))
        error_history.append(grad.release().frobenius_norm())
        if error_history[-1] < eps:
            print('Small grad norm {} is reached at iteration {}'.format(error_history[-1], it))
            return x, it, error_history
        conj_dir_prev, conj_dir = conj_dir,\
                                  conjugate_direction(x_prev, grad_prev, conj_dir_prev, x, grad)
        alpha = minimize_scalar(fun=cost_function, bounds=(0., 10.), method='bounded')['x']
        #alpha_init = closed_form_initial_guess(conj_dir, delta, sigma_set)
        #alpha = line_search(cost_function, xk=alpha_init, )
        m = 0
        for i in range(20):
            if cost_raw(x) - cost_raw(svd_retraction(x + 0.5**m * alpha * conj_dir.release(), r)) >= \
                    -0.0001 * 0.5**m * alpha * grad.release().scalar_product(conj_dir.release()):
                m = i
                break
        print('iter:{}, alpha: {}, m: {} error: {}'.format(it, alpha, m, error_history[-1]))
        x_prev, x = x, svd_retraction(x + 0.5**m * alpha * conj_dir.release(), r)
    print('Error {} is reached at iteration {}. Cannot converge'.format(error_history[-1], maxiter))
    return x, maxiter, error_history