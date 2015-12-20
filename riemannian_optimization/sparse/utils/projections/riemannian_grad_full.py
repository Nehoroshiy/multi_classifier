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
from riemannian_optimization.lowrank_matrix import ManifoldElement


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
