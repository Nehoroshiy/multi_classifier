import numpy as np


def csvd(a, r=None):
    if r is None or r > min(a.shape):
        r = min(a.shape)
    u, s, v = np.linalg.svd(a, full_matrices=False)
    return u[:, :r], s[:r], v[:r]