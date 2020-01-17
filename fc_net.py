from builtins import range
from builtins import object
import numpy as np
 
from layers import *
 
 
class FullyConnectedNet(object):
    """
    A fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of [H, ...], and perform classification over C classes.
 
    The architecure should be like affine - relu - affine - softmax for a one
    hidden layer network, and affine - relu - affine - relu- affine - softmax for
    a two hidden layer network, etc.
 
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
 
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
 
    def __init__(self, input_dim, hidden_dim=[10, 5], num_classes=10,
                 weight_scale=0.1):
        """
        Initialize a new network.
 
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        self.hidden_dim = hidden_dim
        ############################################################################
        # TODO: Initialize the weights and biases of the net. Weights              #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.num_layers = len(hidden_dim)
        # self.weights = {}
        # self.biases = {}
        for i in np.arange(0, self.num_layers + 1, 1):
            #print("initializing,", i)
            if i == 0:
                M1 = input_dim
            else:
                M1 = hidden_dim[i-1]
            if i == self.num_layers:
                N1 = num_classes
            else:
                N1 = hidden_dim[i]
            wkey = "W" + str(i+1)
            bkey = "b" + str(i+1)
            #hkey = "h" + str(i+1)
            self.params[wkey] = np.random.normal(
                0.0, weight_scale, size=(M1, N1))
            self.params[bkey] = np.zeros(shape=(N1,))
            #self.params[hkey] = 0.0
        # print(self.params["W2"])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
 
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
 
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
 
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the net, computing the              #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        X1 = X
        xcache1 = {}
        xcache2 = {}
        for i in np.arange(0, self.num_layers, 1):
            wkey = "W" + str(i+1)
            bkey = "b" + str(i+1)
 
            X1, cache1 = affine_forward(
                X1, self.params[wkey], self.params[bkey])
            xcache1[str(i+1)], dummyw, dummyb = cache1
            #xcache[str(i+1)] = X1
            # = X1
            X1, cache2 = relu_forward(X1)
            xcache2[str(i+1)] = cache2
 
        # we don't apply relu on the last step
        wkey = "W" + str(self.num_layers + 1)
        bkey = "b" + str(self.num_layers + 1)
 
        X1, cache_final = affine_forward(
            X1, self.params[wkey], self.params[bkey])
 
        scores = X1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
 
        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the net. Store the loss            #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k].                                                          #
        ############################################################################
        loss, adx = softmax_loss(scores, y)
        #print("adx:", adx)
        # print(loss)
 
        adx, adw, adb = affine_backward(adx, cache_final)
        grads["W" + str(self.num_layers + 1)] = adw
        grads["b" + str(self.num_layers + 1)] = adb
 
        for i in np.arange(1, self.num_layers + 1, 1):
 
            idx = self.num_layers - i
 
            wkey = "W" + str(idx+1)
            bkey = "b" + str(idx+1)
 
            bi = self.params[bkey]
            wi = self.params[wkey]
            xi = xcache1[str(idx+1)]
            #print("adx size:", adx.shape)
            #print("bi size:", bi.shape)
            #print("wi size:", wi.shape)
            #print("xi size:", xi.shape)
            #
            adx = relu_backward(adx, xcache2[str(idx+1)])
            #
            adx, adw, adb = affine_backward(adx, (xi, wi, bi))
 
            grads[wkey] = adw
            grads[bkey] = adb
 
            #grads[hkey] = adx
            #print(i, X1)
            # print(X1.shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
 
        return loss, grads