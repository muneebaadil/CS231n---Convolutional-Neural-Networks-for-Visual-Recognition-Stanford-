import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      toadd = np.zeros(W.shape) #Gradient matrix to add in every nested iteration. 
      if margin > 0:
        loss += margin
        toadd[:, j] = X[i] #Only jth column of W has incluence in Sj, and 
        #its partial derivative = feature vector of ith training example. 
        toadd[:, y[i]] = toadd[:, y[i]] - X[i] #Similarly, y(i)th column of W 
        #influence and its partial derivative = feature vector of ith training example, 
      dW = dW + toadd #finally subtracting those two to get the jacobian matrix of 
      #jth class, ith training example.
  dW = dW / num_train #Diving the overall gradient matrix by number of training examples. 
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW = dW + 2 * W #Adding the gradient of regularization. 

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  loss = 0.0 
  for i in xrange(num_train): 
    temp = X[i].dot(W) #A vector with jth entry as a score of jth class for ith example
    temp = temp - X[i].dot(W[:, y[i]]) #Subtracting true label's score from each class' score.
    temp = temp + 1 #Adding one as a default geometric margin 
    temp[y[i]] = 0 #Don't need to compute the margin between true class' label. 
    temp = np.maximum(0, temp) #Shifting everything negative to zero. 
    loss += np.sum(temp) #Finally summing over all class' margins to get a total margin for 
    #ith example. 
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W) #Adding the regularization term. 
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
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW