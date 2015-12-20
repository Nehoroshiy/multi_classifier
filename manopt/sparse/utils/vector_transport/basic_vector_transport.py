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