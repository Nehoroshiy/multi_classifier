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

from manopt.lowrank_matrix import ManifoldElement
from manopt.sparse.utils.projections import TangentVector


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
