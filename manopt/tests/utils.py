"""
2015-2016 Constantine Belev const.belev@ya.ru
"""


import scipy as sp

from scipy import sparse


def generate_sigma_set(shape, percent):
    return sp.sparse.random(*shape, density=percent).nonzero()