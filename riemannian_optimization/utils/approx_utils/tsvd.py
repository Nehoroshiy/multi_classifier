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


def rank_chop(s, delta):
    """
    Clip vector s = [\sigma_{1}, \ldots, \sigma{r}] of real values sorted in descending order
    as s without \sigma_{r'}, \sigma_{r'+1} \ldots, \sigma_{r}
    such that \sum\limits_{i=r'}^{r} \sigma_{i} \geqslant \delta
    Parameters
    ----------
    s : (..., r) array_like
        A real-valued array of size r
    delta : float, optional
        clipping parameter. Zero by default, so, if not specified, will cause nothing.
    Returns
    -------
    sr : (..., r') array_like
        Clipped array
    """
    error = 0
    i = 1
    while i <= s.size and error + s[-i] < delta:
        error += s[-i]
        i += 1
    return s.size - i + 1


def tsvd(a, delta=0):
    """
    Approximately factors matrix 'a' as u * np.diag(s) * v
    as SVD without \sigma_{r'}, \sigma_{r'+1} \ldots, \sigma_{r}
    such that \sum\limits_{i=r'}^{r} \sigma_{i} \geqslant \delta
    See also numpy.linalg.svd
    Parameters
    ----------
    a : (..., M, N) array_like
        A real or complex matrix of shape ('M', 'N')
    delta : float, optional
        clipping parameter. Zero by default, so, if not specified, will cause
        ordinary svd execution with economic matrices
    Returns
    -------
    u : (..., M, r') array
        Unitary matrix
    s : (..., K) array
        The clipped singular values for every matrix, sorted in descending order
    v : (..., r', N) array
        Unitary matrix
    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    """
    u, s, v = np.linalg.svd(a, full_matrices=False)
    r_new = rank_chop(s, delta)
    return u[:, :r_new], s[:r_new], v[:r_new]
