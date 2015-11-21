import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
              Softmax loss function, naive implementation (with loops)
              Inputs:
              - W: C x D array of weights
              - X: D x N array of data. Data are D-dimensional columns
              - y: 1-dimensional array of length N with labels 0...K-1, for K classes
              - reg: (float) regularization strength
              Returns:
              a tuple of:
              - loss as single float
              - gradient with respect to weights W, an array of same size as W
              """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = np.dot(W, X)
    shifted = scores - scores.max(0)
    scores_max = scores.max(0)
    scores_argmax = np.argmax(scores, 0)

    #print(scores.shape)
    #print(y.shape)

    for k in xrange(X.shape[1]):
        exp_vec_frac = np.sum(np.exp(shifted[:, k]))
        loss += -shifted[y[k], k] + np.log(exp_vec_frac)
        for i in xrange(W.shape[0]):
            dW[scores_argmax[k], :] -= X[:, k] * np.exp(shifted[i, k]) / exp_vec_frac
            dW[i, :] += X[:, k] * np.exp(shifted[i, k]) / exp_vec_frac
        dW[y[k], :] -= X[:, k]
        dW[scores_argmax[k], :] += X[:, k]
    loss /= y.size
    loss += 0.5 * reg * np.sum(W * W)
    dW /= y.size
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
      Softmax loss function, vectorized version.

      Inputs and outputs are the same as softmax_loss_naive.
      """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[1]
    shifted = np.dot(W, X)
    shifted -= shifted.max(0)
    shifted_exp = np.exp(shifted)
    exp_vec = shifted_exp.sum(0)
    # loss function
    loss = np.average(-shifted[y, np.arange(N)] + np.log(exp_vec))
    loss += 0.5 * reg * np.sum(W * W)
    # dW
    dW[:, :] += np.dot(shifted_exp / exp_vec, X[:, :].T)
    for label in np.arange(dW.shape[0]):
        dW[label, :] -= X[:, y == label].sum(1)
    dW /= y.size
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


class Softmax():
    def __init__(self, learning_rate=1e-6, reg=5e4):
        self.learn_rate = learning_rate
        self.reg = reg

    def fit(self, x, y, init=None, eps=1e-5, batch_size=64):
        self.batch_size = min(batch_size, x.shape[1])
        classes = max(y)
        if init is None:
            init = np.zeros((classes, x.shape[0]))
        self.W = np.array(init, dtype=float)
        loss = 1.
        it = 1
        while (loss > eps):
            indices = np.random.choice(x.shape[1], batch_size, replace=False)
            loss, grad = softmax_loss_vectorized(self.W, self.x[:, indices], self.y[indices])
            self.W -= self.learn_rate * grad
            it += 1
        print('Process converges after {} iterations'.format(it))
        return

    def prob(self, x):
        return np.dot(self.W, x)

    def predict(self, x):
        return np.argmax(self.prob(x), 0)