"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np
from lowrank_matrix import ManifoldElement


def delta_on_sigma_set(x, a, sigma_set):
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


def riemannian_grad_full(x, a, sigma_set, grad=None):
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
    grad = delta_on_sigma_set(x, a, sigma_set) if grad is None else grad
    left_projected = grad.T.dot(x.u)
    right_projected = grad.dot(x.v.T)
    mid = x.u.T.dot(right_projected)
    u = right_projected - x.u.dot(mid)
    v = left_projected - x.v.T.dot(mid.T)

    mid = ManifoldElement(mid, x.r).rdot(x.u).dot(x.v)
    u = ManifoldElement(u, x.r).dot(x.v)
    v = ManifoldElement(v.T, x.r).rdot(x.u)
    return mid + u + v


def riemannian_grad_partial(x, a, sigma_set, grad=None, manifold_elems=False):
    """
    Riemannian gradient as a parts from which one can restore it

    If grad is not given,
    compute projection of Euclidean gradient of function
    $\dfrac{1}{2}| P_{\Sigma}(X - A)|_F^2$ at tangent space to manifold at x.

    Projection at x has the form
    RiemannianGrad f(x) = UMV^* + U_p V^* + U V_p^*,
    where M, U_p and V_p^* are returned by function

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
    out : tuple of ManifoldElements of shapes (M, N)
        matrices M, U_p and V_p^* as partial riemannian gradient
    """
    grad = delta_on_sigma_set(x, a, sigma_set) if grad is None else grad
    left_projected = grad.T.dot(x.u)
    right_projected = grad.dot(x.v.T)
    mid = x.u.T.dot(right_projected)
    u = right_projected - x.u.dot(mid)
    v = left_projected - x.v.T.dot(mid.T)
    if manifold_elems:
        return ManifoldElement(mid, x.r), ManifoldElement(u, x.r), ManifoldElement(v.T, x.r)
    else:
        return mid, u, v.T


def restore_full_from_partial(x, partial):
    """
    Restore full riemannian gradient from it's partial representation
    at ManifoldElement x

    Parameters
    ----------
    x : ManifoldElement, shape (M, N)
        point at which partial gradient was computed
    partial : tuple of ManifoldElements of shapes (M, N)
        matrices M, U_p and V_p^* as partial riemannian gradient

    Returns
    -------
    out : ManifoldElement
        riemannian gradient at x
    """
    mid_proj, u_proj, v_proj = partial
    return mid_proj.rdot(x.u).dot(x.v) + u_proj.dot(x.v) + v_proj.rdot(x.u)


def svd_retraction(x, r):
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
    if isinstance(x, ManifoldElement):
        return ManifoldElement(x, r)
    elif isinstance(x, TangentVector):
        return ManifoldElement(x.release(), r)
    else:
        raise ValueError("Supports only ManifoldElement or TangentVector classes")


#TODO accept also simple matrices
def vector_transport_base(x_base, x, partial):
    """
    Calculates vector transport of a tangent vector's partial representation
    v = UMV^* + U_p V^* + U V_p^*, where x_base = U \Sigma V.

    Parameters
    ----------
    x_base : ManifoldElement, shape (M, N)
        point at manifold from which transport proceed
    x : ManifoldElement, shape (M, N)
        point at manifold to which transport proceed
    partial : tuple of ManifoldElements, shapes (M, N)
        partial representation of a tangent vector v at point x_base

    Returns
    -------
    out : tuple of ManifoldElements, shapes (M, N)
        partial representation of a transported vector v at point x_base
    """
    mid_base, u_base, v_base = partial
    a_v, a_u = ManifoldElement(x_base.v.dot(x.v.T), x.r), ManifoldElement(x_base.u.T.dot(x.u), x.r)
    b_v, b_u = v_base.dot(x.v.T), u_base.T.dot(x.u)
    mid_1, u_1, v_1 = mid_base.rdot(a_u.T).dot(a_v),\
                      mid_base.dot(a_v).rdot(x_base.u),\
                      mid_base.T.dot(a_u).rdot(x_base.v.T)
    mid_2, u_2, v_2 = b_u.T.dot(a_v), u_base.dot(a_v), b_u.rdot(x_base.v.T)
    mid_3, u_3, v_3 = a_u.T.dot(b_v), b_v.rdot(x_base.u), v_base.T.dot(a_u)
    mid = mid_1 + mid_2 + mid_3
    u = u_1 + u_2 + u_3
    u = u - u.rdot(x.u.T).rdot(x.u)
    v = v_1 + v_2 + v_3
    v = v - v.rdot(x.v).rdot(x.v.T)
    v = v.T
    return mid, u, v


class TangentVector():
    """
    TangentVector represents vector from tangent bundle of a rank-r matrix manifold.
    With point x = U \Sigma V^* given, tangent vector writes as
    v = UMV^* + U_p V^* + U V_p^*.
    """
    def __init__(self, base, data):
        if type(base) is not ManifoldElement:
            raise ValueError("base must be ManifoldElement --- point on the manifold")
        self.base = base
        self._released = None
        if type(data) is tuple:
            if len(data) != 3:
                raise ValueError("data must contains 3 ManifoldElements")
            self.m, self.u, self.v = data
            if self.m.shape[0] != self.m.shape[1]\
                    or self.m.shape[0] != self.u.shape[1]\
                    or self.m.shape[1] != self.v.shape[0]:
                raise ValueError("M, U_p and V_p must be rank-r matrices")

    def __neg__(self):
        return TangentVector(self.base, (-self.m, -self.u, -self.v))

    def __add__(self, other, self_base_priority=True):
        if self.base.isclose(other.base):
            return TangentVector(self.base, (ManifoldElement(self.m + other.m, self.base.r),
                                             ManifoldElement(self.u + other.u, self.base.r),
                                             ManifoldElement(self.v + other.v, self.base.r)))
        elif not self_base_priority:
            other_trans = other.transport(self.base)
            return TangentVector(self.base, (ManifoldElement(self.m + other_trans.m, self.base.r),
                                             ManifoldElement(self.u + other_trans.u, self.base.r),
                                             ManifoldElement(self.v + other_trans.v, self.base.r)))
        else:
            self_trans = self.transport(other.base)
            return TangentVector(other.base, (ManifoldElement(self_trans.m + other.m, self.base.r),
                                              ManifoldElement(self_trans.u + other.u, self.base.r),
                                              ManifoldElement(self_trans.v + other.v, self.base.r)))

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if np.isscalar(other):
            return TangentVector(self.base, (self.m * other, self.u * other, self.v * other))
        else:
            raise ValueError("TangentVector supports only multiplication by scalars")

    def __rmul__(self, other):
        if np.isscalar(other):
            return self * other
        else:
            raise ValueError("TangentVector supports only multiplication by scalars")

    def transport(self, base):
        return TangentVector(base, vector_transport_base(self.base, base, (self.m, self.u, self.v)))

    def release(self):
        if self._released is None:
            self._released = restore_full_from_partial(self.base, (self.m, self.u, self.v))
            return restore_full_from_partial(self.base, (self.m, self.u, self.v))
        else:
            return self._released


    @staticmethod
    def zero(base):
        shape, r = base.shape, base.r
        return TangentVector(base, (ManifoldElement(np.zeros((r, r)), r),
                                    ManifoldElement(np.zeros((shape[0], r)), r),
                                    ManifoldElement(np.zeros((r, shape[1])), r)))
