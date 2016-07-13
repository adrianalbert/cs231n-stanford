import numpy as np
from random import shuffle
from scipy.misc import logsumexp

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  for i in range(len(X)):
    score = X[i].dot(W)
    # Z0 = logsumexp(score)
    score -= np.max(score) # f becomes [-666, -333, 0]
    p =  np.exp(score)
    Z = p.sum()
    p /= Z
    loss += - score[y[i]] + np.log(Z)
    g = p[:,np.newaxis] * X[i]
    g[y[i]] = -X[i] * (1 - p[y[i]])
    dW += g.T
  loss /= len(y)
  loss += 0.5 * reg * (W**2).sum()
  dW /= len(y)
  dW += reg * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
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
  score = X.dot(W)
  score -= score.max(1)[:,np.newaxis]
  p = np.exp(score)
  Z = p.sum(1)
  p /= Z[:,np.newaxis]
  loss = -score[range(len(y)),y] + np.log(Z)
  loss = loss.mean() + 0.5 * reg * (W**2).sum()

  idx = np.zeros_like(p)
  idx[range(len(y)),y] = 1
  dW = -X.T.dot(idx - p) / len(y) + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

