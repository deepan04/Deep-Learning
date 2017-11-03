import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs:
  - W: K x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1 # margin of the SVM
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  # 1) Dot product of weight and data matrix
  XW = np.dot(W,X)
  # 2) get correct class scores using y  
  correct_class=XW[y,np.arange(X.shape[1])]
  # 3) find margins by using element wise maximum function
  #print np.matrix(correct_class).shape
  mar=np.maximum(0,XW-np.matrix(correct_class) + delta)
  #print mar.shape
  # Make correct classes 0
  mar[y,np.arange(X.shape[1])]=0
  #print mar.shape
  # get loss by summing and dividing by n
  loss = np.sum(mar)
  loss /= X.shape[1]
  # adjust by regularization strength
  loss += 0.5 * reg * np.sum(np.square(W))
    
  
    
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # create a binary matrix  
  binary_mat=mar
  binary_mat[mar>0]=1
  
  # sum of all incorrect classes  
  #print binary_mat.shape
  sum=np.sum(binary_mat,axis=0)
    
  # y coordinate decreases and hence negative  
  binary_mat[y,np.arange(X.shape[1])]= -sum
    
  dW = (np.dot(binary_mat,X.T))
  dW = dW / X.shape[1]
  dW = dW + reg*W      
  pass

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
