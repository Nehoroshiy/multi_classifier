"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = x.reshape((x.shape[0], -1)).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x.reshape((x.shape[0], -1)).T.dot(dout)
  db = dout.T.dot(np.ones(dout.shape[0]))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.asarray(x).clip(0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout.copy()
  dx[x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']

  ht = 1 + (H + 2 * pad - HH) / stride
  wt = 1 + (W + 2 * pad - WW) / stride
  """
  print(H, HH, ht)
  print(W, WW, wt)
  """
  hrange = np.arange(0, ht)
  wrange = np.arange(0, wt)

  out = np.zeros((N, F, ht, wt))
  for i, X in enumerate(x):
    x_padded = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    for h_idx in hrange:
      for w_idx in wrange:
        h_strided, w_strided = h_idx * stride, w_idx * stride
        for filter_idx in range(F):
          """
          print('*'*20)
          print(h_strided, w_strided)
          print(x_padded[:, h_strided : h_strided + HH, w_strided : w_strided + WW].shape)
          print(w[filter_idx, ...].shape)
          print(b[np.newaxis, np.newaxis, np.newaxis, filter_idx].shape)
          print('-'*20)
          """
          out[i, filter_idx, h_idx, w_idx] = \
          np.sum(x_padded[:, h_strided : h_strided + HH, w_strided : w_strided + WW] * w[filter_idx, ...]) + \
          b[np.newaxis, np.newaxis, np.newaxis, filter_idx]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  # dout (N, F, H', W')
  # - x: Input data of shape (N, C, H, W)
  # - w: Filter weights of shape (F, C, HH, WW)
  # - b: Biases, of shape (F,)
  x, w, b, conv_param = cache
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  pad = conv_param['pad']
  stride = conv_param['stride']

  ht = 1 + (H + 2 * pad - HH) / stride
  wt = 1 + (W + 2 * pad - WW) / stride
  """
  print(H, HH, ht)
  print(W, WW, wt)
  """
  hrange = np.arange(0, ht)
  wrange = np.arange(0, wt)

  dx_padded = np.pad(np.zeros_like(x), ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  dx = dx_padded[:, :, pad: pad + x.shape[2], pad: pad + x.shape[3]]
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  out = np.zeros((N, F, ht, wt))
  for i, X in enumerate(x):
    x_padded = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    for h_idx in hrange:
      for w_idx in wrange:
        db += 1. * dout[i, :, h_idx, w_idx]
        h_strided, w_strided = h_idx * stride, w_idx * stride
        for filter_idx in range(F):
          dx_padded[i, :, h_strided : h_strided + HH, w_strided : w_strided + WW] += dout[i, filter_idx, h_idx, w_idx] * w[filter_idx, ...]
          dw[filter_idx, ...] += dout[i, filter_idx, h_idx, w_idx] * x_padded[:, h_strided : h_strided + HH, w_strided : w_strided + WW]#[:, ::-1, ::-1]
  dx = dx.copy()
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  ph, pw = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  h2, w2 = (H - ph) / stride + 1, (W - pw) / stride + 1
  out = np.zeros((x.shape[0], x.shape[1], h2, w2))
  for h in range(h2):
    for w in range(w2):
      h_strided, w_strided = h * stride, w * stride
      out[:, :, h, w] = x[:, :, h_strided: h_strided + ph, w_strided: w_strided + pw].max(axis=(2, 3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  dx = np.zeros_like(x)
  N, C, H, W = x.shape
  ph, pw = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  h2, w2 = (H - ph) / stride + 1, (W - pw) / stride + 1
  for h in range(h2):
    for w in range(w2):
      h_strided, w_strided = h * stride, w * stride
      x_view = x[:, :, h_strided: h_strided + ph, w_strided: w_strided + pw]
      #max_indices = np.unravel_index(x_view.reshape(x_view.shape[:2] + (-1,)).argmax(-1), (ph, pw))
      ix = np.unravel_index(x_view.reshape(x_view.shape[:2] + (-1,)).argmax(-1).ravel(), x_view.shape[-2:])
      first = (np.repeat(np.arange(N), C), np.tile(np.arange(C), N))
      dx[first + (h_strided + ix[0],) + (w_strided + ix[1],)] = dout[first + (np.repeat(h, N * C),) + (np.repeat(w, N * C),)]
      #for i, c in np.ndindex(dx.shape[:2]):
      #  max_indices = np.unravel_index(x_view[i, c].argmax(), (ph, pw))
      #  indx, indy = max_indices
      #  dx[i, c, h_strided + indx, w_strided + indy] += dout[i, c, h, w]


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


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

