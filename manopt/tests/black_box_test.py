"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np
from manopt.approximator.black_box import BlackBox


def blackbox_test():
    def eq(a, b):
        return np.array_equal(a, b)

    A = np.random.random((10, 10))
    b = BlackBox(A)

    assert eq(b[:, :], A[:, :])
    assert eq(b[:, 1], A[:, 1])
    assert eq(b[:, [1, 2]], A[:, [1, 2]])
    assert eq(b[1, :], A[1, :])
    assert eq(b[[1, 2], :], A[[1, 2], :])
    assert eq(b[1, 2], A[1, 2])
    assert eq(b[[1], [2]], A[[1], [2]])
    assert eq(b[::2, ::3], A[::2, ::3])
    assert eq(b[[1, 2, 3], [1, 2, 3]], A[[1, 2, 3], [1, 2, 3]])
    assert eq(b[1], A[1])


if __name__ == "__main__":
    blackbox_test()
