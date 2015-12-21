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
import scipy as sp
from utils import generate_sigma_set
from scipy.sparse import csr_matrix, coo_matrix

from manopt.approximator.lowrank_matrix import ManifoldElement


def strict_check(manifold_element, original_matrix):
    assert(np.allclose(manifold_element.full_matrix(), original_matrix))


def test_constructor_matrix(shape, r):
    test_matrix = np.random.random(shape)
    manifold_element = ManifoldElement(test_matrix)
    r_manifold_element = ManifoldElement(test_matrix, r)
    strict_check(manifold_element, test_matrix)
    strict_check(ManifoldElement(manifold_element, r), r_manifold_element.full_matrix())


def test_constructor_double(shape, r):
    m, n = shape
    u = np.random.randn(m, r) * (2 * np.random.random() + 1)
    v = np.random.randn(r, n) * (2 * np.random.random() + 1)
    test_matrix = u.dot(v)
    manifold_element = ManifoldElement((u, v))
    r_manifold_element = ManifoldElement((u, v), r)
    strict_check(manifold_element, test_matrix)
    strict_check(r_manifold_element, test_matrix)


def test_constructor_triple(shape, r):
    m, n = shape
    u = np.linalg.qr(np.random.randn(m, r))[0]
    s = np.sort(np.abs(np.random.randn(r)))[::-1]
    v = sp.linalg.rq(np.random.randn(r, n), mode='economic')[1]

    full_matrix = u.dot(np.diag(s)).dot(v)
    r_full_matrix = u[:, :r].dot(np.diag(s[:r])).dot(v[:r, :])
    manifold_element = ManifoldElement((u, s, v))
    r_manifold_element = ManifoldElement((u, s, v), r)
    strict_check(manifold_element, full_matrix)
    strict_check(r_manifold_element, r_full_matrix)


def test_constructor_copy(shape, r):
    r_delta = r - 2
    m, n = shape
    u = np.linalg.qr(np.random.randn(m, r))[0]
    s = np.sort(np.abs(np.random.randn(r)))[::-1]
    v = sp.linalg.rq(np.random.randn(r, n), mode='economic')[1]

    test_matrix = u.dot(np.diag(s)).dot(v)
    shrink_test_matrix = u[:, :r_delta].dot(np.diag(s[:r_delta])).dot(v[:r_delta, :])
    manifold_element = ManifoldElement(test_matrix, r)
    copy_element = ManifoldElement(manifold_element)
    shrink_element = ManifoldElement(manifold_element, r - 2)
    strict_check(manifold_element, copy_element.full_matrix())
    strict_check(shrink_element, shrink_test_matrix)


def test_constructor(shape, r, niter=10):
    for _ in range(niter):
        test_constructor_matrix(shape, r)
        test_constructor_double(shape, r)
        test_constructor_triple(shape, r)
        test_constructor_copy(shape, r)


def test_binary_operation(shape, r, operation='+'):
    permitted_operations = ['+', '-', '*']
    if operation not in permitted_operations:
        raise ValueError(
            'Operation mus be one of those: {}, but it is {}'.format(permitted_operations,
                                                                     operation))
    left_element, right_element = ManifoldElement.rand(shape, r), ManifoldElement.rand(shape, r)
    left_full, right_full = left_element.full_matrix(), right_element.full_matrix()
    strict_check(eval("left_element " + operation + " right_element"),
                 eval("left_full " + operation + " right_full"))


def test_binary_operations(shape, r, niter=10):
    operations = ['+', '-', '*']
    for operation in operations:
        for _ in range(niter):
            test_binary_operation(shape, r, operation)


def test_dot_product(shape, r, niter=10):
    for _ in range(niter):
        left_element = ManifoldElement.rand(shape, r)
        right_element = ManifoldElement.rand(shape[::-1], r)
        left_full, right_full = left_element.full_matrix(), right_element.full_matrix()
        strict_check(left_element.dot(right_element), left_full.dot(right_full))
        strict_check(left_element.dot(right_full), left_full.dot(right_full))
        strict_check(right_element.rdot(left_element), left_full.dot(right_full))
        strict_check(right_element.rdot(left_full), left_full.dot(right_full))


def test_transpose(shape, r, niter=10):
    for _ in range(niter):
        elem = ManifoldElement.rand(shape, r)
        strict_check(elem.transpose(), elem.full_matrix().T)
        strict_check(elem.T, elem.full_matrix().T)


def test_evaluation(shape, r, niter=10):
    for _ in range(niter):
        elem = ManifoldElement.rand(shape, r)
        full = csr_matrix(elem.full_matrix())
        for percent in np.linspace(0.01, 1., 10):
            sigma = generate_sigma_set(shape, percent)
            temp = csr_matrix(coo_matrix((np.array(full[sigma]).ravel(), sigma), shape=shape))
            assert(np.allclose(temp.todense(), elem.evaluate(sigma).todense()))


def test_trace(shape, r, niter=10):
    for _ in range(niter):
        elem = ManifoldElement.rand(shape, r, norm=100 * np.random.random() + 10)
        full = elem.full_matrix()
        assert(np.isclose(elem.trace(), np.trace(full)))


def test_scalar_product(shape, r, niter=10):
    for _ in range(niter):
        left = ManifoldElement.rand(shape, r, norm=100 * np.random.random() + 10)
        right = ManifoldElement.rand(shape, r, norm=100 * np.random.random() + 10)
        left_full = left.full_matrix()
        right_full = right.full_matrix()
        assert(np.isclose(left.scalar_product(right), np.trace(left_full.dot(right_full.T))))


if __name__ == '__main__':
    # shape, r, niter
    args = ((200, 100), 10, 10)

    test_constructor(*args)
    test_binary_operations(*args)
    test_dot_product(*args)
    test_transpose(*args)
    test_evaluation(*args)
    test_scalar_product(*args)

    # equally-shaped test
    sym_args = ((200, 200), 10, 10)

    test_trace(*sym_args)


