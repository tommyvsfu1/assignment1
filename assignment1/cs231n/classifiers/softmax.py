import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = np.dot(X,W)
    exp_sum_denominator = 0.0
    soft_max_i = 0.0
    for j in range(num_classes):
      exp_sum_denominator += np.exp(scores[i,j])
    
    soft_max_i = np.exp(scores[i,y[i]])/exp_sum_denominator
    
    for j in range(num_classes):
      dW_j = np.zeros(X[i].shape)
      if j ==y[i]:
        dW_j = ( np.exp(scores[i,y[i]])*exp_sum_denominator - np.exp(scores[i,y[i]])*np.exp(scores[i,j]) ) * X[i]
      else :
        dW_j = - np.exp(scores[i,y[i]])*np.exp(scores[i,j])  * X[i]
      dW_j /=(exp_sum_denominator) * (exp_sum_denominator)
      multiplier = -1/soft_max_i
      dW_j *= multiplier
      dW[:,j] += dW_j


    loss += (-np.log(soft_max_i))

  # Average
  loss /= num_train

  # Regularization
  loss += reg*np.sum(W*W)
  dW = dW/num_train + 2 * reg * W 

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X,W)  
  exp_scores = np.exp(scores)
  exp_correct_scores = np.exp(scores[np.arange(num_train),y])
  exp_sum_denominator = np.sum(exp_scores,axis=1) # axis=1 means row sum

  soft_max = exp_correct_scores/exp_sum_denominator #shape=(500,)

  loss = -np.sum(np.log(soft_max))



  multiplier = -1  / (exp_sum_denominator*exp_sum_denominator)
  multiplier /= soft_max

  num_train_indexing = np.arange(num_train)
  num_classes_indexing = np.arange(num_classes)
  train_row = multiplier[:,np.newaxis]*X # N x D

  scores_reduce_to_one = -exp_correct_scores[:,np.newaxis]*exp_scores #N x C
  print("scores_reduce_to_one shape",scores_reduce_to_one.shape)

  #scores_reduce_to_one = np.sum(scores_reduce_to_one,axis=0)
  #scores_reduce_to_one = -1*np.sum(scores,axis=0) #shape=(C,)
  #train_row_reduce_to_one = np.sum(train_row,axis=0) #shape=(D,)


  mask = np.zeros((num_train,num_classes)) #shape=N x C
  mask[np.arange(num_train),y]=1.0 
  train_scores = exp_correct_scores*exp_sum_denominator #Nx1
  mask_scores = train_scores[:,np.newaxis]*mask 
  #mask_scores_sum = np.sum(mask_scores,axis=0) #shape=(C,)

  class_scores = scores_reduce_to_one+mask_scores # N x C
  #dW_transpose= class_scores[:,np.newaxis]*train_row_reduce_to_one # shape=(C,D)
  #dW = dW_transpose.T
  # X = N x D
  dW = np.dot(X.T ,class_scores)

  # Average
  loss /= num_train

  # Regularization
  loss += reg*np.sum(W*W)
  dW = dW/num_train + 2 * reg * W  


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

