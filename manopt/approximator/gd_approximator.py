"""
2015-2016 Constantine Belev const.belev@ya.ru
"""


import numpy as np
import scipy as sp
from scipy import sparse, optimize
from lowrank_matrix import ManifoldElement
from approximator_api import AbstractApproximator
from manifold_functions import TangentVector, svd_retraction
from manifold_functions import riemannian_grad_partial, delta_on_sigma_set
from scipy.sparse import linalg, csc_matrix
from scipy.optimize import minimize_scalar

EPS = 1e-9


class GDApproximator(AbstractApproximator):
    def __init__(self):
        AbstractApproximator.__init__(self)
        self.target_matrix = None
        self.density = None
        self.norm_bound = None
        self.sigma_set = None
        self.x = None
        self.grad = None

    def approximate(self, a, r, sigma_set=None, x0=None, maxiter=900, eps=EPS):
        return self._approximate(a, r, sigma_set=sigma_set, x0=x0, maxiter=maxiter, eps=eps)

    def _approximate(self, a, r, sigma_set=None, x0=None, maxiter=900, eps=EPS):
        if a is None:
            raise ValueError("target matrix must be provided")
        self.target_matrix = a
        self.initialization(sigma_set)

        #for rank in range(1, r):
        #    x0, it, err = self.gd_approximate(r=rank, x0=x0,
        #                                      maxiter=50, eps=eps)
        return self.gd_approximate(r=r, x0=x0, maxiter=maxiter, eps=eps)

    def gd_approximate(self, r, x0=None, maxiter=100, eps=1e-9):
        self.init_condition(r, x0)
        error_history = []
        for it in range(maxiter):
            self.step()
            error_history.append(self.loss())
            print('it: %s, error: %s' % (it, error_history[-1]))
            if error_history[-1] < self.norm_bound * eps:
                return self.x, it, error_history
        return self.x, maxiter, error_history

    def step(self):
        self.gd_grad()
        self.gd_step()
        pass

    def init_condition(self, r, x0):
        if x0 is None:
            x0 = ManifoldElement.rand(self.target_matrix.shape, r, norm=self.norm_bound)
        self.x = ManifoldElement(x0, r)
        self.grad = -TangentVector(self.x, riemannian_grad_partial(self.x, self.target_matrix,
                                                               self.sigma_set, manifold_elems=True))
        return None

    def gd_grad(self):
        riemannian_grad = riemannian_grad_partial(self.x, self.target_matrix,
                                                  self.sigma_set, manifold_elems=True)
        self.grad = -TangentVector(self.x, riemannian_grad)
        return None

    def gd_step(self):
        alpha = minimize_scalar(lambda x: self.cost_func(x), bounds=(0., 10.), method='bounded')['x']
        if alpha is None:
            alpha = 1.
        print(alpha)
        self.x = svd_retraction(self.x + alpha * self.grad.release(), self.x.r)
        #self.armijo_backtracking(lambda x: self.cost_raw(x), alpha)[0]
        return None

    def armijo_backtracking(self, alpha, maxiter=20):
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
            x_new = svd_retraction(self.x + (0.5 ** i * alpha) * self.grad.release(), self.x.r)
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

    def cost_func(self, param):
        retracted = svd_retraction(self.x + param * self.grad.release(), self.x.r)
        return self.cost_raw(retracted)
