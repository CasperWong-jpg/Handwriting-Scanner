import numpy as np
from util import *

# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    # See eq (12) and (16) from https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    b = np.zeros(out_size)
    normalized_init_val = np.sqrt(6) / np.sqrt(in_size + out_size)
    W = np.random.uniform(low=-normalized_init_val, high=normalized_init_val, size=(in_size, out_size))

    params['W' + name] = W
    params['b' + name] = b

# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    ones = np.ones(x.shape)
    res = ones / (ones + np.exp(-x))
    return res

def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = np.matmul(X, W) + b
    post_act = activation(pre_act)
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    x_norm = x - np.max(x, axis=1, keepdims=True)
    s = np.exp(x_norm)
    S = np.sum(s, axis=1, keepdims=True)

    return s / S

# compute average loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    D, n = probs.shape
    # Compute loss
    loss = -(1 / D) * np.sum(y * np.log(probs))

    # Compute accuracy – choose highest probability and compare with y
    fx_idx = np.argmax(probs, axis=1)
    y_idx = np.argmax(y, axis=1)
    acc = np.count_nonzero(fx_idx == y_idx) / D
    return loss, acc

# Function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    batchSize, _ = X.shape

    # See derivations here http://16385.courses.cs.cmu.edu/fall2023/lecture/nn/slide_118
    # Mimicking first line of slide. ie. dL/df = delta, df/da = post_act_deriv, da/dw = X (a = XW + b)
    post_act_deriv = activation_deriv(post_act)
    dF_da = delta * post_act_deriv

    grad_W = X.T @ dF_da
    grad_b = dF_da.T @ np.ones((batchSize))
    grad_X = dF_da @ W.T

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    assert(x.shape[0] == y.shape[0])
    populationSize = x.shape[0]

    batchOrder = np.arange(populationSize)
    np.random.shuffle(batchOrder)
    # If populationSize not a multiple of batch_size, last batch will be smaller
    numBatches = np.ceil(populationSize / batch_size).astype(int)
    batches = []
    for i in range(numBatches):
        batch_x = x[batchOrder[batch_size * i: batch_size * (i + 1)]]
        batch_y = y[batchOrder[batch_size * i: batch_size * (i + 1)]]
        batches.append((batch_x, batch_y))
    return batches
