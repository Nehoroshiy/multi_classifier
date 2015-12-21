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
from lowrank_matrix import ManifoldElement
from approximator_api import AbstractApproximator
from manifold_functions import TangentVector, svd_retraction
from manifold_functions import riemannian_grad_partial, delta_on_sigma_set
from scipy.sparse import linalg, csc_matrix


EPS = 1e-9


def closed_form_initial_guess(vec, delta, sigma_set):
    n_mat = csc_matrix(vec.release().evaluate(sigma_set)).T
    trace_first = n_mat.multiply(delta.T).sum()
    trace_second = n_mat.multiply(n_mat).sum()
    return np.abs(trace_first / trace_second)


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
        x_new = svd_retraction(x + (0.5**i * alpha) * conj_direction.release(), x.r)
        bound = (0.5**i * scale) * direction.release().scalar_product(conj_direction.release())
        if func(x) - func(x_new) >= bound:
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


def init_condition(a, sigma_set, r, x0, eps):
    density = 1.0 * len(sigma_set[0]) / np.prod(a.shape)
    norm_bound = np.linalg.norm(np.array(a[sigma_set])) / np.sqrt(density)

    if x0 is None:
        x0 = ManifoldElement.rand(a.shape, r, norm=norm_bound)
    x_, x = ManifoldElement(x0, r), ManifoldElement(x0, r)

    grad = -TangentVector(x0, riemannian_grad_partial(x0, a, sigma_set, manifold_elems=True))
    conj_, conj = TangentVector.zero(x), TangentVector.zero(x)
    return x_, x, conj_, conj, grad, eps * norm_bound


def cg_grad(x, a, sigma_set, grad):
    riemannian_grad = riemannian_grad_partial(x, a, sigma_set, manifold_elems=True)
    grad_, grad = grad, -TangentVector(x, riemannian_grad)
    return grad_, grad


def cg_step(x_, conj_, grad_, x, a, sigma_set, conj, grad):
    conj_, conj = conj, conjugate_direction(x_, grad_, conj_, x, grad)

    alpha = closed_form_initial_guess(conj, delta_on_sigma_set(x, a, sigma_set), sigma_set)
    x_, x = x, armijo_backtracking(lambda x: cost_raw(a, x, sigma_set), x, alpha, grad, conj)[0]
    return conj_, conj, x_, x


def cg_fuck(a, sigma_set, r, x0=None, maxiter=900, eps=1e-9):
    x_, x, conj_, conj, grad, delta = init_condition(a, sigma_set, r, x0, eps)

    error_history = []

    for it in range(maxiter):
        grad_, grad = cg_grad(x, a, sigma_set, grad)
        conj_, conj, x_, x = cg_step(x_, conj_, grad_, x, a, sigma_set, conj, grad)

        error = grad.release().frobenius_norm()
        error_history.append(error)

        print('iter:{}, error: {}'.format(it, error_history[-1]))

        if error < delta:
            return x, it, error_history

    return x, maxiter, error_history


class CGApproximator(AbstractApproximator):
    def __init__(self):
        AbstractApproximator.__init__(self)
        self.target_matrix = None
        self.density = None
        self.norm_bound = None
        self.sigma_set = None
        self.x_prev, self.x = None, None
        self.grad_prev, self.grad = None, None
        self.conj_prev, self.conj = None, None

    def error(self):
        return self.grad.release().frobenius_norm()

    def approximate(self, a, r, sigma_set=None, x0=None, maxiter=900, eps=EPS):
        return self._approximate(a, r, sigma_set=sigma_set, x0=x0, maxiter=900, eps=eps)

    def _approximate(self, a, r, sigma_set=None, x0=None, maxiter=900, eps=EPS):
        if a is None:
            raise ValueError("target matrix must be provided")
        self.target_matrix = a
        self.initialization(sigma_set)


        for rank in range(1, r):
            x0, it, err = self.cg_approximate(r=rank, x0=x0,
                                              maxiter=10, eps=eps)
        return self.cg_approximate(r=r, x0=x0, maxiter=maxiter, eps=eps)
        """
        for rank in range(1, r):
            x0, it, err = cg_fuck(a, sigma_set, rank, x0=x0, maxiter=10, eps=eps)
        return cg_fuck(a, sigma_set, r, x0=x0, maxiter=maxiter, eps=eps)
        """
    def initialization(self, sigma_set=None):
        if sigma_set is None:
            self.sigma_set = self.target_matrix.nonzero()
        else:
            self.sigma_set = sigma_set
        self.sigma_set[0][:] = self.sigma_set[0][self.sigma_set[1].argsort(kind='mergesort')]
        self.sigma_set[1][:] = self.sigma_set[1][self.sigma_set[1].argsort(kind='mergesort')]
        self.sigma_set[1][:] = self.sigma_set[1][self.sigma_set[0].argsort(kind='mergesort')]
        self.sigma_set[0][:] = self.sigma_set[0][self.sigma_set[0].argsort(kind='mergesort')]

        self.density = 1.0 * len(self.sigma_set[0]) / np.prod(self.target_matrix.shape)
        part_norm = np.linalg.norm(np.array(self.target_matrix[self.sigma_set]))
        self.norm_bound = part_norm / np.sqrt(self.density)
        print('est. norm: %s' % self.norm_bound)
        return None

    def cg(self, a, sigma_set, r, x0=None, maxiter=900, eps=1e-9):
        x_, x, conj_, conj, grad, delta = init_condition(a, sigma_set, r, x0, eps)

        error_history = []

        for it in range(maxiter):
            grad_, grad = cg_grad(x, a, sigma_set, grad)
            conj_, conj, x_, x = cg_step(x_, conj_, grad_, x, a, sigma_set, conj, grad)

            error = grad.release().frobenius_norm()
            error_history.append(error)

            print('iter:{}, error: {}'.format(it, error_history[-1]))

            if error < delta:
                return x, it, error_history

        return x, maxiter, error_history

    def cg_approximate(self, r, x0=None, maxiter=100, eps=1e-9):
        self.x_prev, self.x, self.conj_prev, self.conj, self.grad, delta = \
            init_condition(self.target_matrix, self.sigma_set, r, x0=x0, eps=eps)
        error_history = []
        for it in range(maxiter):
            self.grad_prev, self.grad = cg_grad(self.x, self.target_matrix, self.sigma_set, self.grad)
            self.conj_prev, self.conj, self.x_prev, self.x =\
                cg_step(self.x_prev, self.conj_prev, self.grad_prev, self.x,
                        self.target_matrix, self.sigma_set, self.conj, self.grad)
            error = self.grad.release().frobenius_norm()
            error_history.append(error)
            print('iter:{}, error: {}'.format(it, error_history[-1]))

            if error < delta:
                return self.x, it, error_history
        return self.x, maxiter, error_history

    def cg_approximate_old(self, r, x0=None, maxiter=100, eps=1e-9):
        self.init_condition(r, x0)
        error_history = []
        for it in range(maxiter):
            self.step()
            error_history.append(self.error())
            print('it: %s, error: %s' % (it, error_history[-1]))
            if error_history[-1] < self.norm_bound * eps:
                return self.x, it, error_history
        return self.x, maxiter, error_history

    def step(self):
        self.cg_grad()
        self.cg_step()
        pass

    def init_condition(self, r, x0):
        if x0 is None:
            x0 = ManifoldElement.rand(self.target_matrix.shape, r, norm=self.norm_bound)
        self.x_prev, self.x = ManifoldElement(x0, r), ManifoldElement(x0, r)
        self.grad = -TangentVector(self.x, riemannian_grad_partial(self.x, self.target_matrix,
                                                               self.sigma_set, manifold_elems=True))
        self.grad_prev = self.grad
        self.conj_prev, self.conj = TangentVector.zero(self.x), TangentVector.zero(self.x)
        return None

    def cg_grad(self):
        riemannian_grad = riemannian_grad_partial(self.x, self.target_matrix,
                                                  self.sigma_set, manifold_elems=True)
        self.grad_prev, self.grad = self.grad, -TangentVector(self.x, riemannian_grad)
        return None

    def cg_step(self):
        self.conj_prev, self.conj = self.conj, self.conjugate_direction()

        alpha = closed_form_initial_guess(self.conj,
                                          delta_on_sigma_set(self.x,
                                                             self.target_matrix,
                                                             self.sigma_set),
                                          self.sigma_set)
        self.x_prev, self.x = \
            self.x, self.armijo_backtracking(lambda x: self.cost_raw(x), alpha)[0]
        return None

    def armijo_backtracking(self, func, alpha, maxiter=20):
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
            x_new = svd_retraction(self.x + (0.5 ** i * alpha) * self.conj.release(), self.x.r)
            bound = (0.5 ** i * scale) * self.grad.release().scalar_product(self.conj.release())
            if self.cost_raw(self.x) - self.cost_raw(x_new) >= bound:
                return x_new, 0.5 ** i * scale
        return x_new, 0.5 ** maxiter * scale

    def cost_raw(self, elem):
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
        return 0.5 * sp.sparse.linalg.norm(elem.evaluate(self.sigma_set) - self.target_matrix) ** 2

    def conjugate_direction(self):
        grad_prev_trans = self.grad_prev.transport(self.x)
        conj_prev_trans = self.conj_prev.transport(self.x)

        delta = self.grad - grad_prev_trans
        beta = max(0, delta.release().scalar_product(self.grad.release()) /
                   self.grad_prev.release().frobenius_norm() ** 2)
        conj = -self.grad + beta * conj_prev_trans

        angle = self.grad.release().scalar_product(conj.release()) / \
                np.sqrt(conj.release().frobenius_norm() ** 2 *
                        self.grad.release().frobenius_norm() ** 2)
        if angle <= 0.1:
            conj = self.grad
        return conj
