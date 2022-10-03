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


def step(z):
    return 1 * (z > 0)


def linear(z):
    return z


# def vector_to_matrices(parameters, connections, x):
#     """reshape 1D vector into appropriate matrices"""
#     L = len(parameters) // 2
#     num_wts = np.cumsum([0] + connections)
#     for ell in range(1, L + 1):
#         shape = parameters["W" + str(ell)].shape
#         parameters["W" + str(ell)] = x[num_wts[ell - 1] : num_wts[ell]].reshape(shape)
#     return parameters


def vector_to_matrices(parameters, connections, biases, x):
    """reshape 1D vector into appropriate matrices"""
    L = len(parameters) // 2
    num_wts = np.cumsum([0] + connections)
    num_bias = np.cumsum([np.sum(connections)] + biases)
    for ell in range(1, L + 1):
        shape = parameters["W" + str(ell)].shape
        bshape = parameters["b" + str(ell)].shape
        parameters["W" + str(ell)] = x[num_wts[ell - 1] : num_wts[ell]].reshape(shape)
        parameters["b" + str(ell)] = x[num_bias[ell - 1] : num_bias[ell]].reshape(
            bshape
        )
    return parameters


# def intitialize_layer_params(layer_dims, seed):
#     """initialize weights and bias' based on layer dimensions"""
#     params = {}
#     L = len(layer_dims)
#     rng = np.random.default_rng(seed)
#     for ell in range(1, L):
#         params["W" + str(ell)] = (
#             rng.normal(size=(layer_dims[ell], layer_dims[ell - 1])) * 0.01
#         )
#         params["b" + str(ell)] = np.zeros((layer_dims[ell], 1))
#     return params


def intitialize_layer_params(layer_dims, seed, init="uniform"):
    """initialize weights and bias' based on layer dimensions"""
    params = {}
    L = len(layer_dims)
    rng = np.random.default_rng(seed)
    init_func = uniform_wt

    if init == "glorot":
        init_func = glorot
    elif init == "glorot_norm":
        init_func = glorot_norm
    elif init == "he":
        init_func = he

    for ell in range(1, L):
        params["W" + str(ell)] = init_func(layer_dims[ell], layer_dims[ell - 1], rng)
        params["b" + str(ell)] = np.zeros((layer_dims[ell], 1))
    return params


def uniform_wt(m, n, rng):
    """Uniform weight initliazation"""
    # m = nodes in current layer
    # n = nodes in previous layer
    return rng.uniform(size=(m, n))


def glorot(m, n, rng):
    """Glorot weight initliazation - tanh or sigmoid"""
    # m = nodes in current layer
    # n = nodes in previous layer
    lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))
    return lower + rng.uniform(size=(m, n)) * (upper - lower)


def glorot_norm(m, n, rng):
    """Normalized Glorot weight initliazation - tanh or sigmoid"""
    # m = nodes in current layer
    # n = nodes in previous layer
    lower, upper = -(np.sqrt(6.0) / np.sqrt(n + m)), (np.sqrt(6.0) / np.sqrt(n + m))
    return lower + rng.uniform(size=(m, n)) * (upper - lower)


def he(m, n, rng):
    """He weight initialization - ReLU"""
    # m = nodes in current layer
    # n = nodes in previous layer
    std = np.sqrt(2.0 / n)
    return rng.normal(size=(m, n)) * std


# def linear_forward(X, parameters, afunc):
#     """forward pass through network"""
#     A = X
#     L = len(parameters) // 2
#     for ell in range(1, L):
#         A_prev = A
#         W = parameters["W" + str(ell)]
#         b = parameters["b" + str(ell)]
#         A = afunc(np.dot(A_prev, W.T))  # not considering bias term
#     WL = parameters["W" + str(L)]
#     bL = parameters["b" + str(L)]
#     AL = linear(np.dot(A, WL.T))  # not considering bias term
#     AL = NormalScoreTransformer().transform(AL.squeeze())
#     return AL


def linear_forward(X, parameters, afunc):
    """forward pass through network"""
    A = X
    L = len(parameters) // 2
    for ell in range(1, L):
        A_prev = A
        W = parameters["W" + str(ell)]
        b = parameters["b" + str(ell)]
        A = afunc(np.dot(A_prev, W.T) + b.T)
    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    AL = linear(np.dot(A, WL.T) + bL.T)
    AL = NormalScoreTransformer().transform(AL.squeeze())
    return AL


def node_variance(X, parameters, afunc):
    """calculate variance of node outputs for sensitivity testing"""
    out_var = {}
    A = X
    L = len(parameters) // 2
    for ell in range(1, L):
        A_prev = A
        W = parameters["W" + str(ell)]
        b = parameters["b" + str(ell)]
        A = afunc(np.dot(A_prev, W.T) + b.T)
        out_var["W" + str(ell)] = np.var(A, axis=0)
    WL = parameters["W" + str(L)]
    bL = parameters["b" + str(L)]
    AL = linear(np.dot(A, WL.T) + bL.T)
    out_var["W" + str(L)] = np.var(AL, axis=0)
    return out_var

