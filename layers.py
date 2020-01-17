from builtins import range
import numpy as np
 
 
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
 
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
 
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
 
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.     #
 
    # re-shaping x
    shapey = x.shape[0]  # N
    new_x = x.reshape((shapey, -1))
 
    out = np.matmul(new_x, w) + b
 
    ###########################################################################
    # pass      comment out pass right ?
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
      - b: Biases, of shape (M,)
 
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # print("x shape:", x.shape)
    # print("w shape:", w.shape)
    # print("b shape:", b.shape)
    # print("dout shape:", dout.shape)
 
    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)
 
    shapey = x.shape[0]
    new_x = x.reshape((shapey, -1))
    dw = np.dot(new_x.T, dout)
 
    db = np.dot(dout.T, np.ones(x.shape[0]))
 
    # pass  comment out pass right ?
    # print("returning")
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout  # np.array(dout)
    # if x <= 0:
    #    dx = dx * 0
    dx[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
 
 
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
 
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
 
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
 
    loss = 0.0
    dx = None
    ###########################################################################
    # TODO: Implement the softmax loss                                        #
    ###########################################################################
 
    shapey = y.shape[0]
    shapex = x.shape[0]
    maxes = np.max(x, axis=1)  # maxes is a vector of each row's maximum
 
    tihelp = np.zeros(shape=x.shape)
    for i in np.arange(0, shapex, 1):
        tihelp[i] = np.exp(x[i] - maxes[i])
 
    # term1 is the -(s_c - m) term from HW pdf file
    term1 = np.zeros(shape=shapex)
    for i in np.arange(0, len(term1), 1):
        term1[i] = maxes[i] - x[i, y[i]]
 
    rowsum = np.sum(tihelp, axis=1)
    term2 = np.log(rowsum)
 
    ll = term1 + term2
 
    loss = np.sum(ll) / shapey
 
    dx = np.zeros(shape=x.shape)
    #denom = np.zeros(shape=shapex)
    for i in np.arange(0, shapex, 1):
        #tihelp[i] = np.exp(x[i])
        #denom[i] = np.sum(tihelp[i])
        dx[i] = tihelp[i] / rowsum[i]
 
    dx[np.arange(shapex), y] -= 1
    dx /= shapex
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx