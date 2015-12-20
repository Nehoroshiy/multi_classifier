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

from ..loss_functions import delta_on_sigma_set
from manopt.lowrank_matrix import ManifoldElement


def riemannian_grad_partial(x, a, sigma_set, grad=None, as_manifold_elements=False):
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
    if as_manifold_elements:
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
