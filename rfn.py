#!/usr/bin/python
'''
Python wrapper for librfn.

Copyright © 2015-2017 Thomas Unterthiner
Additional Contributions by Thomas Adler, Balázs Bencze

Licensed under GPL, version 2 or a later (see LICENSE.txt)
'''

import os
import time
import ctypes as ct
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import sparse


import sys
if sys.version_info < (3,):
    range = xrange


_curdir = os.path.dirname(os.path.realpath(__file__))
_librfn = ct.cdll.LoadLibrary(os.path.join(_curdir, 'librfn.so'))
_default_gpu_id = -1
_use_cpu_id = -2

_librfn.calculate_W.argtypes = [
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.float32),
    ct.c_int, ct.c_int, ct.c_int,
    ct.c_int, ct.c_int, ct.c_float,
    ct.c_int]


_librfn.train_rfn.restype = ct.c_int
_librfn.train_rfn.argtypes = [
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.float32),
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
    ct.c_float, ct.c_float, ct.c_float, ct.c_float,
    ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float,
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
    ct.c_int
]


_librfn.calculate_W_sparse.argtypes = [
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.int32),
    np.ctypeslib.ndpointer(np.int32),
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.float32),
    ct.c_int, ct.c_int, ct.c_int,
    ct.c_int, ct.c_int, ct.c_float,
    ct.c_int]


_librfn.train_rfn_sparse.restype = ct.c_int
_librfn.train_rfn_sparse.argtypes = [
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.int32),
    np.ctypeslib.ndpointer(np.int32),
    np.ctypeslib.ndpointer(np.float32),
    np.ctypeslib.ndpointer(np.float32),
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
    ct.c_float, ct.c_float, ct.c_float, ct.c_float,
    ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float,
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
    ct.c_int
]


_input_noise_types = {"dropout": 1, "saltpepper": 2, "gaussian": 3}
_activation_types = {"linear": 0, "relu": 1, "leaky": 2, "sigmoid": 3, "tanh": 4}

def train_rfn(X, n_hidden, n_iter, etaW, etaP, minP, dropout_rate,
              input_noise_rate=0.0, startP=0.1, startW=None,
              l2_weightdecay=0.0, l1_weightdecay=0.0,
              input_noise_type="saltpepper", activation="relu",
              h_threshold=0.0, momentum=0.0, applyNewtonUpdate=True,
              batch_size=-1, seed=None, gpu_id="default"):
    '''Trains a Rectified Factor Network (RFN).

    Trains an RFN as explained in
    "Rectified Factor Networks", Clevert et al., NIPS 2015

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Input samples

    n_hidden : int
        Number of latent variables to estimate

    n_iter : int
        Number of iterations to run the algorithm

    etaW : float
        Learning rate of the W parameter

    etaP : float
        Learning rate of the Psi parameter
        (It's probably save to set this to the same value as etaW)

    minP : float
        Minimal value for Psi. Should be in 1e-8 - 1e-1

    dropout_rate : float in [0, 1)
        Dropout rate for the latent variables

    input_noise_rate : float
        Noise/dropout rate for input variables

    startW : array-like, shape = (n_hidden, n_features)
        Optional pre-initialized weights parameters. Useful if one wants to
        continue training of an old result.

    l2_weightdecay : float
        L2 penalty for weight decay

    l2_weightdecay : float
        L1 penalty for weight decay

    input_noise_type : one of 'dropout', 'saltpepper' or 'gaussian'
        Type of input noise

    activation : one of ('linear', 'relu', 'leaky', 'sigmoid', 'tanh')
        Activation function for hidden/latent variables.

    h_threshold : float
        Threshhold for rectifying/leaky activations

    momentum : float
        Momentum term for learning

    applyNewtonUpdate : boolean
        Whether to use a Newton update (default) or a Gradient Descent step.

    batch_size : int
        If > 2, this will activate mini-batch learning instead of full
        batch learning.

    seed : int
        Seed for the random number generator

    gpu_id : int or "cpu"
        ID of the gpu device to use. If set to "cpu", the calculations will
        be performed on the CPU instead.


    Returns
    -------
    A tuple of three elements:

    W : array-like, shape = (n_hidden, n_features)
        The weight matrix W used in the paper, used to transform the
        hidden/latent variables back to visibles.
    Psi : array-like, shape = (n_features, )
        Variance of each input feature dimension (Psi in the paper's formulas)
    Wout : array-like, shape = (n_hidden, n_features)
        Weight matrix needed to transform the visible variables back into
        hidden variables. Normally this is done via
            `H = np.maximum(0, np.dot(Wout, X.T))`
    '''

    if seed is None:
        # should be different for each call on each process
        seed = np.uint32(hash(os.getpid() + time.time()) % 4294967295)
    if gpu_id == "default":
        gpu_id = _default_gpu_id
    elif gpu_id == "cpu":
        gpu_id = _use_cpu_id

    rng = np.random.RandomState(seed)
    if startW is None:
        W = rng.normal(scale=0.01, size=(n_hidden, X.shape[1])).astype(np.float32)
    else:
        W = startW.astype(np.float32, order="C")
    if isinstance(startP, np.ndarray):
        P = startP.astype(np.float32)
    else:
        P = np.array([startP] * X.shape[1], dtype=np.float32)

    Wout = np.empty((W.shape[0], W.shape[1]), np.float32)

    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
        if not X.has_canonical_format:
            X = X.sorted_indices()
        _librfn.train_rfn_sparse(X.data, X.indices, X.indptr, W, P, X.shape[0], X.shape[1], n_hidden, n_iter,
                          batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                          l2_weightdecay, l1_weightdecay, momentum, _input_noise_types[input_noise_type],
                          _activation_types[activation], 1, applyNewtonUpdate, seed, gpu_id)
        _librfn.calculate_W_sparse(X.data, X.indices, X.indptr, W, P, Wout,
                                X.shape[0], X.shape[1], W.shape[0],
                                _activation_types[activation], 1, h_threshold,
                                gpu_id)
    else:
        X = X.astype(np.float32, order="C")
        _librfn.train_rfn(X, W, P, X.shape[0], X.shape[1], n_hidden, n_iter,
                          batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                          l2_weightdecay, l1_weightdecay, momentum, _input_noise_types[input_noise_type],
                          _activation_types[activation], 1, applyNewtonUpdate, seed, gpu_id)
        _librfn.calculate_W(X, W, P, Wout,
                                X.shape[0], X.shape[1], W.shape[0],
                                _activation_types[activation], 1, h_threshold,
                                gpu_id)

    return W, P, Wout


from sklearn.base import BaseEstimator, TransformerMixin
class RectifiedFactorNetwork(BaseEstimator, TransformerMixin):
    '''Implements a sklearn interface for RFN.'''
    def __init__(self, n_hidden=128, n_iter=50, etaW=0.1, etaP=0.1, minP=1e-2, dropout_rate=0.0,
              input_noise_rate=0.0, startP=0.1, startW=None,
              l2_weightdecay=0.0, l1_weightdecay=0.0,
              input_noise_type="saltpepper", activation="relu",
              h_threshold=0.0, momentum=0.0, applyNewtonUpdate=True,
              batch_size=-1, seed=None, gpu_id="default"):
        self.n_hidden = n_hidden
        self.n_iter = n_iter
        self.etaW = etaW
        self.etaP = etaP
        self.minP = minP
        self.dropout_rate = dropout_rate
        self.input_noise_rate = input_noise_rate
        self.startP = startP
        self.startW = startW
        self.l2_weightdecay = l2_weightdecay
        self.l1_weightdecay = l1_weightdecay
        self.input_noise_type = input_noise_type
        self.activation = activation
        self.h_threshold = h_threshold
        self.momentum = momentum
        self.applyNewtonUpdate = applyNewtonUpdate
        self.batch_size = batch_size
        self.seed = seed
        self.gpu_id = gpu_id

    def fit(self, x, y=None):
        res = train_rfn(x, self.n_hidden, self.n_iter, self.etaW, self.etaP, self.minP, self.dropout_rate,
              self.input_noise_rate, self.startP, self.startW,
              self.l2_weightdecay, self.l1_weightdecay,
              self.input_noise_type, self.activation,
              self.h_threshold, self.momentum, self.applyNewtonUpdate,
              self.batch_size, self.seed, self.gpu_id)
        self.w, self.psi, self.wout = res
        return self

    def transform(self, x):
        h = np.maximum(np.dot(x, self.wout.T), 0)
        s = h.std(1)
        s[s < 1e-6] = 1
        h /= s[:, None]  ## TODO: should I really scale the h?
        return h

    def inverse_transform(self, h):
        r = np.dot(h, self.w)
        return r
