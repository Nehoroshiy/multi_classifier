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

from manopt.lowrank_matrix import ManifoldElement
from ..vector_transport import vector_transport_base
from riemannian_grad_partial import restore_full_from_partial


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
        return restore_full_from_partial(self.base, (self.m, self.u, self.v))

    @staticmethod
    def zero(base):
        shape, r = base.shape, base.r
        return TangentVector(base, (ManifoldElement(np.zeros((r, r)), r),
                                    ManifoldElement(np.zeros((shape[0], r)), r),
                                    ManifoldElement(np.zeros((r, shape[1])), r)))
