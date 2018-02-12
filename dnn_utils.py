import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation function in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A --  output of sigmoid of Z, same shape as Z
    cache -- returns Z as well, useful for backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, Z


def relu(Z):
    """
    Implement the ReLU function in numpy

    Arguments:
    Z -- Output of the linear layer of any shape

    Returns:
    A -- Post-activation parameter of the same shape as Z
    cache -- a python dictionary containing 'A'; stored for computing back prop
    """

    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)

    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for a single sigmoid unit

    Arguments:
    dA -- post activation gradient, of any shape
    cache -- 'Z' where we store for computing back prop efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert(dZ.shape == Z.shape)
    return dZ


def relu_backward(dA, cache):
    """
    Implements the backward prop for a single ReLU

    Arguments:
    dA -- post activation gradient, of any shape
    cache -- 'Z' where we store for computing back prop efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ
