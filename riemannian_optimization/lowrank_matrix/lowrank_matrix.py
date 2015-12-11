import numpy as np
import scipy as sp

from scipy import linalg

from riemannian_optimization.utils.approx_utils import csvd, tsvd


def orth(a):
    return np.allclose(a.T.dot(a), np.eye(a.shape[1]))


def torth(a):
    return np.allclose(a.dot(a.T), np.eye(a.shape[0]))


def indices_unveil(self, indices):
    indices = np.asarray(indices)
    return np.vstack([np.repeat(indices[0], indices[1].size),
               np.tile(indices[1], indices[0].size)]).T


class ManifoldElement(object):
    def __init__(self, data, r=None):
        if type(data) == ManifoldElement:
            if r is None:
                # copy constructor
                self.u = data.u.copy()
                self.s = data.s.copy()
                self.v = data.v.copy()
                self.shape = data.shape
                self.r = data.r
                return
            elif r <= data.r:
                self.u = data.u[:, :r].copy()
                self.s = data.s[:r].copy()
                self.v = data.v[:r, :].copy()
                self.shape = data.shape
                self.r = r
                return
            else:
                self.u = np.zeros(data.shape[0], r)
                self.s = np.zeros(r)
                self.v = np.zeros(r, data.shape[1])
                self.u[:, data.r] = data.u
                self.s[:data.r] = data.s
                self.v[:data.r, :] = data.v
                self.shape = data.shape
                self.r = r
                return
        elif type(data) == np.ndarray:
            self.r = min(data.shape) if r is None else min(min(data.shape), r)
            self.shape = data.shape
            self.u, self.s, self.v = csvd(data, self.r)
            return
        elif type(data) == tuple:
            if len(data) == 2:
                # we have u, v matrices
                u, v = data
                if u.shape[1] != v.shape[0]:
                    raise ValueError('u, v must be composable, but have shapes {}, {}'.format(u.shape, v.shape))
                self.r = u.shape[1] if r in None else min(u.shape[1], r)
                rt, qt = sp.linalg.rq(v, mode='economic')
                u = u.dot(rt)
                self.shape = (u.shape[0], v.shape[1])
                self.u, self.s, v = csvd(u, self.r)
                """
                if np.linalg.norm(self.s) == 0:
                    self.u = np.zeros(self.shape[0]).reshape((-1, 1))
                    self.s = np.zeros(1)
                    self.v = np.zeros(self.shape[1]).reshape((1, -1))
                """
                if np.linalg.norm(self.s) == 0:
                    self.u = np.zeros((self.shape[0], self.r)).reshape((-1, self.r))
                    self.s = np.zeros(self.r)
                    self.v = np.zeros((self.r, self.shape[1])).reshape((self.r, -1))
                self.v = v.dot(qt[:self.r, :])
                return
            elif len(data) == 3:
                # we have u, s, v^t factorization (numpy-like)
                # + maybe we have lost non-increasing property of s (due to sum operator)
                # and we need to perform rearrange
                u, s, v = data
                if len(s.shape) == 2:
                    s = np.diag(s)
                if not orth(u) or not torth(v):
                    raise ValueError('u and v must be orthogonal as SVD factors')
                if u.shape[1] != v.shape[0] or u.shape[1] != s.size:
                    raise ValueError('u, s, v must be svd factorization of some matrix')
                self.r = u.shape[1] if r is None else min(r, u.shape[1])
                self.shape = (u.shape[0], v.shape[1])
                self.u, self.s, self.v = data
                if self.r < u.shape[1]:
                    self.u = self.u[:, :self.r].copy()
                    self.s = self.s[:self.r].copy()
                    self.v = self.v[:self.r, :].copy()
                if not np.allclose(self.s.argsort(), np.arange(self.s.size)[::-1]):
                    # we need to rearrange svd decomposition
                    self.rearrange()
                return
            else:
                raise ValueError("Arguments in tuple are not supported")
        else:
            raise ValueError("Arguments are not supported")

    def rearrange(self):
        # rearrange restores SVD decomposition of A + B
        # when we have [U_A U_B] [\Sigma_A 0 \n 0 \Sigma_B] [V_A V_B]^*
        # block representation of sum
        # we perform sort of singular values and permute U and V factors
        # according to such a permutation
        permutation = self.s.argsort()
        self.u = self.u[:, permutation]
        self.v = self.v[permutation, :]
        self.s = self.s[permutation]
        return

    def __add__(self, other):
        if type(other) == np.ndarray:
            other = ManifoldElement(other, self.r)
            return self.__add__(other)
        elif type(other) == ManifoldElement:
            u = np.hstack((self.u, other.u))
            s = np.concatenate((self.s, other.s))
            v = np.vstack((self.v, other.v))
            u, r = np.linalg.qr(u)
            rt, v = sp.linalg.rq(v, mode='economic')
            middle_factor = r.dot(np.diag(s)).dot(rt)
            u_mid, s, v_mid = np.linalg.svd(middle_factor, full_matrices=False)
            u = u.dot(u_mid)
            v = v_mid.dot(v)

            factorization = (u, s, v)
            return ManifoldElement(factorization)
        else:
            raise ValueError("operation is not supported for ManifoldElement and {}".format(type(other)))

    def __radd__(self, other):
        if type(other) not in [np.ndarray, ManifoldElement]:
            raise ValueError("operation is not supported for ManifoldElement and {}".format(type(other)))
        # because of addition commutativity
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) == np.ndarray:
            other = ManifoldElement(other, self.r)
            return self.__sub__(other)
        elif type(other) == ManifoldElement:
            return self + (-other)
        else:
            raise ValueError("operation is not supported for ManifoldElement and {}".format(type(other)))

    def __rsub__(self, other):
        if type(other) not in [np.ndarray, ManifoldElement]:
            raise ValueError("operation is not supported for ManifoldElement and {}".format(type(other)))
        # because of addition commutativity
        return self.__add__(other)

    def __neg__(self):
        return ManifoldElement((-self.u, self.s, self.v))

    def __mul__(self, other):
        if np.isscalar(other):
            return ManifoldElement((self.u, self.s * other, self.v))
        # TODO u and v factors are not orthogonal!
        assert self.shape == other.shape
        u = np.zeros((self.shape[0], self.r * other.r))
        for i in range(self.shape[0]):
            u[i, :] = np.kron(self.u[i, :], other.u[i, :])
        s = np.kron(self.s, other.s)
        v = np.zeros((self.r * other.r, self.shape[1]))
        for i in range(self.shape[1]):
            v[:, i] = np.kron(self.v[:, i], other.v[:, i])
        return ManifoldElement((u, s, v))

    def __rmul__(self, other):
        if np.isscalar(other):
            return ManifoldElement((self.u, self.s * other, self.v))
        else:
            raise NotImplementedError('rmul operator is not implemented, except for scalars')

    def dot(self, other):
        if type(other) not in [np.ndarray, ManifoldElement]:
            raise ValueError("operation not supported for ManifoldElement and {}".format(type(other)))
        r_factor = np.dot(np.diag(self.s), self.v.dot(other.u)).dot(np.diag(other.s))
        u, s, v = np.linalg.svd(r_factor, full_matrices=False)
        u = self.u.dot(u)
        v = v.dot(other.v)
        return ManifoldElement((u, s, v))

    def full_matrix(self):
        return self.u.dot(np.diag(self.s)).dot(self.v)

    def frobenius_norm(self):
        return np.linalg.norm(self.s)


class MatrixLowRank(object):
    def __init__(self, data, r=None):
        if type(data) == np.ndarray:
            if len(data.shape) != 2:
                raise ValueError('supports only 2d arrays')
            self.r = min(data.shape) if r is None else r
            self.shape = data.shape
            self.u, s, self.v = csvd(data, self.r)
            self.norm = np.linalg.norm(s)
            s /= self.norm
            self.v = np.dot(np.diag(s), self.v)
        if type(data) == tuple:
            if len(data) == 2:
                u, v = data
                # we have u, v matrices
                if (u.shape[1] != v.shape[0]):
                    raise ValueError('u, v must be multiplicable, but have shapes {}, {}'.format(u.shape, v.shape))
                self.r = u.shape[1] if r is None else r
                rt, qt = sp.linalg.rq(v, mode='economic')
                u = u.dot(rt)
                self.shape = (u.shape[0], v.shape[1])
                self.u, s, v = csvd(u, self.r)
                self.norm = np.linalg.norm(s)
                if self.norm == 0:
                    self.v = np.zeros_like(qt)
                    return
                s /= self.norm
                self.v = np.diag(s).dot(np.dot(v, qt))
            else:
                raise ValueError('supports only a = uv matrix factorization')

    def __add__(self, other):
        if type(other) != MatrixLowRank:
            raise ValueError('second summand must be matrix_lowrank')
        r = self.r + other.r
        return MatrixLowRank((np.hstack([self.u, other.u]), np.vstack([self.norm * self.v, other.norm * other.v])), r)

    def __radd__(self, other):
        if type(other) != MatrixLowRank:
            raise ValueError('second summand must be matrix_lowrank')
        return other.__add__(self)

    def __sub__(self, other):
        if type(other) != MatrixLowRank:
            raise ValueError('second summand must be matrix_lowrank')
        r = self.r + other.r
        return MatrixLowRank((np.hstack([self.u, other.u]), np.vstack([self.norm * self.v, -other.norm * other.v])), r)

    def __rsub__(self, other):
        if type(other) != MatrixLowRank:
            raise ValueError('second summand must be matrix_lowrank')
        return other.__sub__(self)

    def indices_unveil(self, indices):
        indices = np.asarray(indices)
        return np.vstack([np.repeat(indices[0], indices[1].size),
                   np.tile(indices[1], indices[0].size)]).T

    def __getitem__(self, indices):
        assert len(indices) == 2, 'indices must be 2 dimensional'
        indx, indy = indices
        if np.isscalar(indx):
            # two scalar indices
            assert np.isscalar(indy)
            return np.dot(self.u[indx] * self.s, self.v[:, indy])
        else:
            # two vector indices
            assert len(indx) == len(indy)


    def round(self, eps=1e-8):
        rt, qt = sp.linalg.rq(self.v)
        u = self.u.dot(rt)
        self.u, s, v = tsvd(u, delta=np.linalg.norm(u)*eps)
        self.r = s.size
        self.norm *= np.linalg.norm(s)
        s /= np.linalg.norm(s)
        self.v = np.diag(s).dot(v.dot(qt))

    def full_matrix(self):
        return np.dot(self.u, self.norm * self.v)