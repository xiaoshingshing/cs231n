import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_data = X.shape[0]
  for i in xrange(num_data):
    scores = X[i,:].dot(W)
    scores = np.exp(scores - np.max(scores))
    total = np.sum(scores)
    for j in xrange(num_classes):
      mul = scores[j]/total
      if y[i]==j:
        mul -= 1
      dW[:,j] += mul * X[i,:]
       
    frac = scores[y[i]] / total
    loss += -np.log(frac)
  loss /= num_data
  loss += reg * np.sum(W * W)
  dW /= num_data
  dW += 2 * reg * W

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
  num_data = X.shape[0]
  scores = X.dot(W)
  scores = np.exp(scores - np.max(scores, axis=1)[:,np.newaxis])
  correct_scores = scores[np.arange(num_data),y]
  frac = correct_scores / np.sum(scores, axis=1)
  loss = -np.sum(np.log(frac)) / num_data + reg*np.sum(W*W)

  scores = scores / np.sum(scores, axis=1)[:,np.newaxis]
  scores[np.arange(num_data), y] -= 1
  dW = X.T.dot(scores)
  dW /= num_data
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

