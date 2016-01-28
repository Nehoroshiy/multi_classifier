"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np
from manopt import ManifoldElement


def frobenius_norm(x):
    if isinstance(x, ManifoldElement):
        return x.frobenius_norm()
    else:
        return np.linalg.norm(x)


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
    We multiply this against a weight matrix of shape (D, M) where
    D = \prod_i d_i

    Inputs:
    x - Input data, of shape (N, d_1, ..., d_k)
    w - Weights, of shape (D, M), ManifoldElement
    b - Biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M), ManifoldElement
    - cache: (x, w, b)
    """
    out = None
    X = ManifoldElement(x) if not isinstance(x, ManifoldElement) else x
    out = X.dot(w) + ManifoldElement.row_stack(b, x.shape[0])
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M), ManifoldElement
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M), ManifoldElement
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = dout.dot(w.T)
    XT = ManifoldElement(x).T
    dw = XT.dot(dout)
    db = dout.sum(0)
    return dx, dw, db


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
