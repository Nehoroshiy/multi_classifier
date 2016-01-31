"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np
import scipy as sp

from scipy import linalg, sparse

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix


def csvd(a, r=None):
    if r is None or r > min(a.shape):
        r = min(a.shape)
    u, s, v = np.linalg.svd(a, full_matrices=False)
    return u[:, :r], s[:r], v[:r]


def orth(a, eps=1e-12):
    return np.linalg.norm(a.T.dot(a) - np.eye(a.shape[1])) < eps * np.sqrt(a.shape[1])


class ManifoldElement(object):
    """
    ManifoldElement(data, r=None)

    Returns a rank-constrained SVD-based approximation of matrix, given by 2D array, or one of
    two decompositions: $A = UV^*$ or $A =  U\SigmaV^*$, where \Sigma is diagonal

    Parameters
    ----------
    data : array_like or (U, V) or (U, s, V) tuple
        matrix, which approximation we want to find
    r : int
        rank constraint


    Examples
    --------
    a = np.random.random((10, 5))
    approximator = ManifoldElement(a, r=4)
    approx_full = approximator.full_matrix()
    print(np.linalg.norm(a - approx_full) / np.linalg.norm(a))

    """
    def __init__(self, data, r=None):
        if isinstance(data, ManifoldElement):
            if r is None:
                self.__init__(data, data.r)
            elif r <= data.r:
                self.u = data.u[:, :r].copy()
                self.s = data.s[:r].copy()
                self.v = data.v[:r, :].copy()
                self.shape = data.shape
                self.r = r
            else:
                self.u = np.zeros((data.shape[0], r))
                self.s = np.zeros(r)
                self.v = np.zeros((r, data.shape[1]))
                self.u[:, :data.r] = data.u
                self.s[:data.r] = data.s
                self.v[:data.r, :] = data.v
                self.shape = data.shape
                self.r = data.r
        elif isinstance(data, np.ndarray):
            self.r = min(data.shape) if r is None else min(min(data.shape), r)
            self.shape = data.shape
            self.u, self.s, self.v = csvd(data, self.r)
        elif isinstance(data, tuple):
            if len(data) == 2:
                # we have u, v matrices
                u, v = data
                if u.shape[1] != v.shape[0]:
                    raise ValueError(
                        'u, v must be composable, but have shapes {}, {}'.format(u.shape, v.shape))
                self.r = u.shape[1] if r is None else min(u.shape[1], r)
                self.shape = (u.shape[0], v.shape[1])
                self.u, self.s, self.v = u, np.ones(self.r), v
                self.balance()
            elif len(data) == 3:
                u, s, v = data
                if len(s.shape) == 2:
                    s = np.diag(s)
                if u.shape[1] != v.shape[0] or u.shape[1] != s.size:
                    raise ValueError('u, s, v must be svd factorization of some matrix')
                self.r = u.shape[1] if r is None else min(r, u.shape[1])
                self.shape = (u.shape[0], v.shape[1])
                self.u, self.s, self.v = data
                if self.r < u.shape[1]:
                    self.u = self.u[:, :self.r].copy()
                    self.s = self.s[:self.r].copy()
                    self.v = self.v[:self.r, :].copy()
                if not (np.diff(self.s) <= 0).all():
                    self.rearrange()        # We lost ordering of s while performing addition
                self.balance()              # We lost orthogonality while performing hadamard
            else:
                raise ValueError("Arguments in tuple are not supported")
        else:
            raise ValueError("Arguments are not supported")

    def rearrange(self):
        """
        Rearrange permutes U \Sigma V^* matrix to obtain
        non-increasing property on diagonal \Sigma matrix

        Returns
        -------
        None
        """
        permutation = self.s.argsort()
        self.u = self.u[:, permutation]
        self.v = self.v[permutation, :]
        self.s = self.s[permutation]
        return None

    def _balance_right(self):
        """
        We only have non-orthogonal v factor, so we need to orthogonalize it

        Returns
        -------
        None
        """
        u, self.s, self.v = csvd(np.diag(self.s).dot(self.v))
        self.u = self.u.dot(u)
        return None

    def _balance_left(self):
        """
        We only have non-orthogonal u factor, so we need to orthogonalize it

        Returns
        -------
        None
        """
        self.u, self.s, v = csvd(self.u.dot(np.diag(self.s)))
        self.v = v.dot(self.v)
        return None

    def _balance(self):
        """
        All factors are non-orthogonal, so we need full orthogonalization

        Returns
        -------
        None
        """
        mid, self.v = sp.linalg.rq(self.v, mode='economic')
        self.u = self.u.dot(np.diag(self.s).dot(mid))
        self.u, self.s, v = csvd(self.u)
        self.v = v.dot(self.v)
        return None

    def balance(self):
        """
        Performs reorthogonalization of factors, if it needs

        Returns
        -------
        None
        """
        #self._balance()

        left_balanced = orth(self.u)
        right_balanced = orth(self.v.T)
        if left_balanced:
            if right_balanced:
                pass
            else:
                self._balance_right()
        else:
            if right_balanced:
                self._balance_left()
            else:
                self._balance()

        #assert(self.is_valid())
        return None

    def is_valid(self):
        """
        Performs validness check. This include checks like:
        * is left core (U) orthogonal?
        * is right core (V^*) orthogonal?
        * is singular values (\Sigma) are in non-increasing order?
        Result will be logical and of these checks.

        Returns
        -------
        status : bool
            Validity status of ManifoldElement object
        """
        left_balanced = orth(self.u)
        right_balanced = orth(self.v.T)
        sigma_sorted = (np.diff(self.s) <= 0).all()
        return left_balanced and right_balanced and sigma_sorted

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            other = ManifoldElement(other, self.r)
            return self.__add__(other)
        elif isinstance(other, ManifoldElement):
            u = np.hstack((self.u, other.u))
            s = np.concatenate((self.s, other.s))
            v = np.vstack((self.v, other.v))
            return ManifoldElement((u, s, v))
        else:
            raise ValueError(
                "operation is not supported for ManifoldElement and {}".format(type(other)))

    def __radd__(self, other):
        if not isinstance(other, np.ndarray) or not isinstance(other, ManifoldElement):
            raise ValueError(
                "operation is not supported for ManifoldElement and {}".format(type(other)))
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            other = ManifoldElement(other, self.r)
            return self.__sub__(other)
        elif isinstance(other, ManifoldElement):
            return self + (-other)
        else:
            raise ValueError(
                "operation is not supported for ManifoldElement and {}".format(type(other)))

    def __rsub__(self, other):
        if not isinstance(other, np.ndarray) or not isinstance(other, ManifoldElement):
            raise ValueError(
                "operation is not supported for ManifoldElement and {}".format(type(other)))
        return self.__sub__(other)

    def __neg__(self):
        return ManifoldElement((-self.u, self.s, self.v))

    def __mul__(self, other):
        if np.isscalar(other):
            return ManifoldElement((self.u, self.s * other, self.v))
        if isinstance(other, ManifoldElement):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not equal.".format(self.shape, other.shape))
            u = np.zeros((self.shape[0], self.r * other.r))
            for i in range(self.shape[0]):
                u[i, :] = np.kron(self.u[i, :], other.u[i, :])
            s = np.kron(self.s, other.s)
            v = np.zeros((self.r * other.r, self.shape[1]))
            for i in range(self.shape[1]):
                v[:, i] = np.kron(self.v[:, i], other.v[:, i])
            return ManifoldElement((u, s, v))
        else:
            raise ValueError("mul supports only scalars and ManifoldElement instances")

    def __rmul__(self, other):
        # TODO do we need this right-side operator?
        # if yes, make it more clever
        if np.isscalar(other):
            return ManifoldElement((self.u, self.s * other, self.v))
        else:
            raise NotImplementedError('rmul operator is not implemented, except for scalars')

    def transpose(self):
        return ManifoldElement((self.v.T, self.s, self.u.T))

    @property
    def T(self):
        return self.transpose()

    def dot(self, other):
        """
        Matrix product.
        Supports np.ndarray, sp.sparse.spmatrix or ManifoldElement left side.

        Parameters
        ----------
        other : sp.sparse.spmatrix, np.ndarray or ManifoldElement
            matrix to perform matrix product

        Returns
        -------
        prod : ManifoldElement
            matrix product
        """
        if self.shape[1] != other.shape[0]:
            raise ValueError("shapes must match!")
        if isinstance(other, ManifoldElement):
            r_mid = np.dot(np.diag(self.s), self.v.dot(other.u)).dot(np.diag(other.s))
            u, s, v = np.linalg.svd(r_mid, full_matrices=False)
            u = self.u.dot(u)
            v = v.dot(other.v)
            return ManifoldElement((u, s, v))
        elif isinstance(other, np.ndarray):
            # TODO there are some ways to optimize (all balance work gone to constructor)
            return ManifoldElement((self.u, self.s, self.v.dot(other)))
        elif sp.sparse.issparse(other):
            return ManifoldElement((self.u, self.s, csc_matrix(other).__rmul__(self.v)))
        else:
            raise ValueError("argument is not supported")

    def rdot(self, other):
        """
        Matrix product where self is right-sided.
        Supports np.ndarray, sp.sparse.spmatrix or ManifoldElement left side.

        Parameters
        ----------
        other : sp.sparse.spmatrix, np.ndarray or ManifoldElement
            matrix to perform matrix product

        Returns
        -------
        prod : ManifoldElement
            matrix product
        """
        if self.shape[0] != other.shape[1]:
            raise ValueError("shapes must match!")
        if isinstance(other, ManifoldElement):
            return other.dot(self)
        elif isinstance(other, np.ndarray):
            return ManifoldElement((other.dot(self.u), self.s, self.v))
        elif sp.sparse.issparse(other):
            return ManifoldElement((csr_matrix(other).dot(self.u), self.s, self.v))
        else:
            raise ValueError("argument is not supported")

    def full_matrix(self):
        """
        Retrieve full matrix from it's decomposition

        Returns
        -------
        a : np.ndarray
            full matrix, given by approximation
        """
        return self.u.dot(np.diag(self.s)).dot(self.v)

    def trace(self):
        """
        Returns trace of a matrix

        Returns
        -------
        out: float
            trace of a matrix
        """
        return np.einsum('ij, ji ->', self.u * self.s, self.v)

    def scalar_product(self, other):
        """
        Returns scalar product of two ManifoldElements

        Parameters
        ----------
        other : ManifoldElement
            element for which scalar product will be computed

        Returns
        -------
        out: float
            scalar product of two ManifoldElements
        """
        if self.shape != other.shape:
            raise ValueError('Shapes mismatch.')
        return self.dot(other.T).trace()

    def frobenius_norm(self):
        """
        Frobenius norm of matrix approximation

        Returns
        -------
        n : float
            norm of the matrix
        """
        return np.linalg.norm(self.s)

    def evaluate(self, sigma_set):
        """
        Retrieve values of full matrix only on a given set of indices

        Parameters
        ----------
        sigma_set : array_like

        Returns
        -------
        out : csr_matrix
            full matrix evaluated at sigma_set
        """
        rows = self.u[sigma_set[0], :] * self.s
        cols = self.v[:, sigma_set[1]]
        data = (rows * cols.T).sum(1)
        assert(data.size == len(sigma_set[0]))
        return csr_matrix(coo_matrix((data, tuple(sigma_set)), shape=self.shape))

    def isclose(self, other, tol=1e-9):
        """
        Check if element is close to another in frobenius norm.

        Parameters
        ----------
        other : ManifoldElement
            element to check closeness
        tol : float
            tolerance for closeness

        Returns
        -------
        status: bool
            flag indicates that to elements are close or not
        """
        if not isinstance(other, ManifoldElement):
            raise ValueError("we can measure closeness only between ManifoldElements")
        largest_norm = max(self.frobenius_norm(), other.frobenius_norm())
        if largest_norm == 0:
            return True
        else:
            return (self - other).frobenius_norm() / largest_norm < tol

    @staticmethod
    def rand(shape, r, norm=None):
        """
        Generates ManifoldElement with parameters:
        * U (shape[0], r) with elements picked from N(0, 1)
        * \Sigma with sorted elements picked from N(0, 1), with desired norm, if given
        * V^* (r, shape[1]) with elements picked from N(0, 1)
        where N(0, 1) means standard normal distribution

        Parameters
        ----------
        shape : tuple of ints
            Output shape.
        r : int
            Desired rank
        norm : float, optional
            Desired frobenius norm of matrix

        Returns
        -------
            out : ManifoldElement
                Randomly generated matrix
        """
        m, n = shape

        u = np.linalg.qr(np.random.randn(m, r))[0]
        s = np.sort(np.abs(np.random.randn(r)))[::-1]
        if norm is not None:
            s *= (norm / np.linalg.norm(s))
        v = sp.linalg.rq(np.random.randn(r, n), mode='economic')[1]
        return ManifoldElement((u, s, v))

    @staticmethod
    def zeros(shape, r):
        u = np.zeros((shape[0], r))
        u[np.diag_indices(r)] = 1.0
        s = np.zeros(r)
        v = np.zeros((r, shape[1]))
        v[np.diag_indices(r)] = 1.0
        elem = ManifoldElement.rand(shape, r)
        elem.u = u
        elem.s = s
        elem.v = v
        return elem

    @staticmethod
    def column_stack(column, n):
        if np.linalg.norm(column) == 0:
            return ManifoldElement.zeros((column.size, n), r=1)
        u = column.reshape((column.size, 1)) / np.linalg.norm(column)
        s = np.array([np.linalg.norm(column)])
        v = np.ones(n).reshape((1, n))
        return ManifoldElement((u, s, v))

    @staticmethod
    def row_stack(row, m):
        if np.linalg.norm(row) == 0:
            return ManifoldElement.zeros((m, row.size), r=1)
        u = np.ones(m, dtype=row.dtype).reshape((m, 1))
        s = np.array([np.linalg.norm(row)])
        v = row.reshape((1, row.size)) / np.linalg.norm(row)
        return ManifoldElement((u, s, v))

    def randomize_last(self):
        if self.r < 1:
            return None
        self.u[:, self.r - 1] = np.random.randn(self.shape[0])
        self.v[self.r - 1] = np.random.randn(self.shape[1])
        self.s[-1] = 1.0 if self.r == 1 else self.s[-2]/10
        return None

    def sum(self, axis=None):
        if axis is None or axis == (0, 1):
            left = np.ones((1, self.shape[0]))
            right = np.ones((self.shape[1], 1))
            return (left.dot(self.u) * self.s).dot(self.v.dot(right))[0, 0]
        elif axis == 0 or -2:
            left = np.ones((1, self.shape[0]))
            return ((left.dot(self.u) * self.s).dot(self.v))[0]
        elif axis == 1 or -1:
            right = np.ones((self.shape[1], 1))
            return (self.u * self.s).dot(self.v.dot(right))[:, 0]
        else:
            raise ValueError('axis must be 0, 1 or tuple')

    def copy(self):
        return ManifoldElement(self, self.r)
