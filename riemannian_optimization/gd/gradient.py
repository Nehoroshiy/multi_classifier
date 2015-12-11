import numpy as np
import scipy as sp

from scipy import linalg, optimize

from numpy.linalg import svd, qr, norm, matrix_rank as mr
from scipy.linalg import rq
from scipy.optimize import minimize_scalar

from riemannian_optimization.utils.approx_utils import csvd
from riemannian_optimization.lowrank_matrix import ManifoldElement


def euclid_grad(x, a, sigma_set):
    """
    :param x: ManifoldElement with rank r
    :param a: matrix to approximate
    :param sigma_set: set of indices where a can be computed exactly
    :return:
    """
    assert x.shape == a.shape, 'x and a must have same dimensions'
    grad = np.zeros(x.shape)
    grad[sigma_set] = x.full_matrix()[sigma_set] - a[sigma_set]
    return grad

def riemannian_grad(x, a, sigma_set, grad=None):
    """
    :param x: ManifoldElement with rank r
    :param a: Matrix-like object that can be computed on set of indices
    :param sigma_set: set of indices where a can be computed exactly
    :param r: rank of approximation
    :return: projection of function P_{\Sigma}(x - a) onto the tangent space of r-rank manifold of at x
    """
    grad = euclid_grad(x, a, sigma_set) if grad is None else grad
    left_projetor, right_projector = x.u.dot(x.u.T), x.v.T.dot(x.v)
    projection = np.dot(left_projetor, grad) + \
                 np.dot(grad, right_projector) - \
                 np.dot(left_projetor, np.dot(grad, right_projector))
    # projection = grad - (np.eye(x.shape[0]) - np.dot(u, u.T)).dot(grad).dot(np.eye(x.shape[1]) - np.dot(v.T, v))
    return projection


def retract(x, r):
    """
    return x from tangent space to rank-r manifold by truncated SVD
    :param x:
    :return: ManifoldElement representation of rank-r projection of x
    """
    return ManifoldElement(x, r)


def approximate(a, sigma_set, r, maxiter=900, eps=1e-9, func=None):
    m, n = a.shape

    # generate random rank-r matrix
    ux = qr(np.random.randn(m, r))[0]
    sx = np.sort(np.abs(np.random.randn(r)))[::-1]
    vx = rq(np.random.randn(r, n), mode='economic')[1]
    x = ManifoldElement((ux, sx, vx))
    err = []

    for it in range(maxiter):
        # get riemannian projection
        grad = euclid_grad(x, a, sigma_set)
        err.append(np.linalg.norm(grad))
        #print(grad)
        #print(x.full_matrix() - a)
        if np.linalg.norm(grad) < eps:
            print('Small grad norm {} is reached at iteration {}'.format(np.linalg.norm(grad), it))
            return it, x, err
        proj = ManifoldElement(-riemannian_grad(x, a, sigma_set))
        #print('Projection of gradient at iteration {}'.format(it))
        #print(proj.full_matrix())
        # line minimization
        def cost_gen(x, a, sigma_set):
            def func(alpha):
                temp = x + alpha * proj
                return 0.5 * np.linalg.norm(temp.full_matrix()[sigma_set] - a[sigma_set])**2
            return func
        alpha = minimize_scalar(fun=cost_gen(x, a, sigma_set), bounds=(0., 10.), method='bounded')['x']
        print('alpha: {}, err: {}'.format(alpha, np.linalg.norm(grad)))
        x_next = x + alpha * proj
        #print('delta at iteration {}'.format(it))
        #print(x_next.full_matrix() - a)

        # retraction
        #print(x_next.shape)
        x_next = retract(x_next, r)

        # stop criteria
        fx = np.linalg.norm(x.full_matrix()[sigma_set] - a[sigma_set])**2
        #print(x_next.shape)
        #print(x_next.full_matrix().shape)
        #print(sigma_set[0].max(), sigma_set[1].max())
        fxn = np.linalg.norm(x_next.full_matrix()[sigma_set] - a[sigma_set])**2
        #if fxn < eps:
        #    # we reach desired accuracy
        #    print('Error {} is reached at iteration {}'.format(fxn, it))
        #    return it, x_next
        x = x_next
    
    print('Error {} is reached at iteration {}. Cannot converge'.format(fxn, it))
    return it, x_next, err