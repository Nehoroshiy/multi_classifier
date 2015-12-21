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

    def approximate(self, a, r, sigma_set=None, x0=None, maxiter=900, eps=EPS):
        return self._approximate(a, r, sigma_set=sigma_set, x0=x0, maxiter=maxiter, eps=eps)

    def _approximate(self, a, r, sigma_set=None, x0=None, maxiter=900, eps=EPS):
        if a is None:
            raise ValueError("target matrix must be provided")
        self.target_matrix = a
        self.initialization(sigma_set)

        for rank in range(1, r):
            x0, it, err = self.cg_approximate(r=rank, x0=x0,
                                              maxiter=20, eps=eps)
        return self.cg_approximate(r=r, x0=x0, maxiter=maxiter, eps=eps)

    def cg_approximate(self, r, x0=None, maxiter=100, eps=1e-9):
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
