import numpy as np
from nscore import NormalScoreTransformer


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def softplus(z):
    return np.log(1 + np.exp(z))


def linear(z):
    return z


def vector_to_matrices(parameters, connections, x):
    """reshape 1D vector into appropriate matrices"""
    L = len(parameters) // 2
    num_wts = np.cumsum([0] + connections)
    for ell in range(1, L + 1):
        shape = parameters["W" + str(ell)].shape
        parameters["W" + str(ell)] = x[num_wts[ell - 1] : num_wts[ell]].reshape(shape)
    return parameters


def intitialize_layer_params(layer_dims, seed):
    """initialize weights and bias' based on layer dimensions"""
    params = {}
    L = len(layer_dims)
    rng = np.random.default_rng(seed)
    for ell in range(1, L):
        params["W" + str(ell)] = (
            rng.normal(size=(layer_dims[ell], layer_dims[ell - 1])) * 0.01
        )
        params["b" + str(ell)] = np.zeros((layer_dims[ell], 1))
    return params


def linear_forward(X, parameters, afunc):
    """forward pass through network"""
    A = X
    L = len(parameters) // 2
    for ell in range(1, L):
        A_prev = A
        W = parameters["W" + str(ell)]
        A = afunc(np.dot(A_prev, W.T))  # not considering bias term
    WL = parameters["W" + str(L)]
    AL = linear(np.dot(A, WL.T))  # not considering bias term
    AL = NormalScoreTransformer().transform(AL.squeeze())
    return AL
