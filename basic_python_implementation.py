#!/usr/bin/python
"""
Implements the RFN algorithm as easily understandable code.

Copyright © 2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.txt)

Contains a very basic CPU and a GPU implementation that is easy to understand.
This code is meant as an instructional ressource, and not suited for production
runs.

The GPU implementation assumes that scikits.cuda.linalg works properly
(which in turn requires CULA). Also, this requires the current development
version of scikits.cuda (as of 2014-08-11).
"""

import time
import numpy as np
from scikits.cuda import linalg as la
import pycuda.curandom as curand
import pycuda.gpuarray as gpu
import pycuda.elementwise as el
import pycuda.driver as drv

from pycuda.compiler import SourceModule
from pycuda.tools import DeviceMemoryPool
from scikits.cuda.cublas import cublasSgemv
from pycuda.elementwise import ElementwiseKernel
from pycuda import cumath

_dropout_kernel = None
_saltpepper_kernel = None
_rng_state = None
_rng_blocks = 128
_rng_threads = 128

_mempool = DeviceMemoryPool()

def init_rng(seed):
    global _dropout_kernel, _saltpepper_kernel, _rng_state, _rng_threads, _rng_blocks
    from pycuda.characterize import sizeof
    ds = sizeof("curandState", "#include <curand_kernel.h>")
    _rng_state = drv.mem_alloc(_rng_threads * _rng_blocks * ds)

    src = SourceModule(
    '''
    #include <curand_kernel.h>

    extern "C"
    {
    __global__ void setup_rng(curandState* rng_state, const unsigned seed)
    {
        const unsigned tid = blockIdx.x*blockDim.x+threadIdx.x;
        curand_init(seed, tid, 0, &rng_state[tid]);
    }

    __global__ void dropout_eltw(float* x, const unsigned size,
                                 float dropout_rate,
                                 curandState* rng_state) {
        const unsigned tid = blockIdx.x*blockDim.x+threadIdx.x;
        const unsigned num_threads = gridDim.x*blockDim.x;
        curandState localState = rng_state[tid];
        for (unsigned i = tid; i < size; i += num_threads)
            x[i] = (curand_uniform(&localState) < dropout_rate) ? 0.0 : x[i];
        rng_state[tid] = localState;
    }

    __global__ void saltpepper_eltw(float* x, const unsigned size,
                                    float dropout_rate,
                                    curandState* rng_state) {
        const unsigned tid = blockIdx.x*blockDim.x+threadIdx.x;
        const unsigned num_threads = gridDim.x*blockDim.x;
        curandState localState = rng_state[tid];
        for (unsigned i = tid; i < size; i += num_threads)
            x[i] = (curand_uniform(&localState) < dropout_rate) ? 0.0 : x[i];
            x[i] = (curand_uniform(&localState) < dropout_rate) ? 1.0 : x[i];
        rng_state[tid] = localState;
    }
    }
    ''', no_extern_c=True)
    setup_rng = src.get_function("setup_rng")
    setup_rng.prepare("Pi")
    setup_rng.prepared_call((_rng_threads, 1, 1), (_rng_blocks, 1, 1),
                            _rng_state, np.uint32(seed))
    _dropout_kernel = src.get_function("dropout_eltw")
    _dropout_kernel.prepare("PifP")
    _saltpepper_kernel = src.get_function("saltpepper_eltw")
    _saltpepper_kernel.prepare("PifP")


def dropout(X, dropout_rate):
    _dropout_kernel.prepared_call((_rng_threads, 1, 1), (_rng_blocks, 1, 1),
        X.gpudata, np.prod(X.shape), np.float32(dropout_rate), _rng_state)
    return X


def saltpepper_noise(X, dropout_rate):
    _saltpepper_kernel.prepared_call((_rng_threads, 1, 1), (_rng_blocks, 1, 1),
        X.gpudata, np.prod(X.shape), np.float32(dropout_rate), _rng_state)
    return X

_unitvariance_step1_kernel = ElementwiseKernel(
    "float* X, float* mean, float* Xsq, const unsigned height",
    "float tmp = X[i] - mean[i % height]; Xsq[i] = tmp*tmp;")

_unitvariance_step2_kernel = ElementwiseKernel(
    "float* work1, const unsigned k",
    "work1[i] = (work1[i] > 1e-7) ? rsqrtf(work1[i]) : 1.0;")

_unitvariance_step3_kernel = ElementwiseKernel(
    "float* X, float* mean, const unsigned height",
    "X[i] *= mean[i % height];")

def to_unit_variance(H):
    ''' Scales H so that column has a variance of 1. '''
    from scikits.cuda.misc import _global_cublas_handle as cublas_handle
    ones = gpu.empty((H.shape[0], 1), np.float32, allocator=_mempool.allocate)
    ones.fill(1.0)
    Hsq = gpu.empty(H.shape, np.float32, allocator=_mempool.allocate)
    mean = gpu.empty((1, H.shape[1]), np.float32, allocator=_mempool.allocate)
    cublasSgemv(cublas_handle, "n", H.shape[1], H.shape[0],
                1.0/H.shape[0], H.gpudata,  H.shape[1], ones.gpudata,
                1, 0.0, mean.gpudata, 1)
    _unitvariance_step1_kernel(H, mean, Hsq, H.shape[1])
    cublasSgemv(cublas_handle, "n", Hsq.shape[1], H.shape[0],
                1.0/H.shape[0], Hsq.gpudata, H.shape[1], ones.gpudata,
                1, 0.0, mean.gpudata, 1)
    _unitvariance_step2_kernel(mean, H.shape[1])
    _unitvariance_step3_kernel(H, mean, H.shape[1])
    return H


def calculate_H_gpu(X, W, P):
    WPW = la.add_diag(P, la.dot(W, W, "t", "n"))
    tmp = la.dot(W, la.inv(WPW, overwrite=True))
    H = la.dot(X, tmp, "n", "t")
    H = gpu.maximum(H, 0)
    H = to_unit_variance(H)
    return H, tmp


def train_rfn_gpu(X, n_hidden, n_iter, learnrateW, learnratePsi, dropout_rate, input_droput_rate, minPsi=0.1, seed=32):
    k = n_hidden
    n, m = X.shape
    W = np.random.normal(scale=0.01, size=(k, m)).astype(np.float32)
    P = np.array([0.1] * m, dtype=np.float32)
    XXdiag = np.diag(np.dot(X.T, X) / n).copy() # explicit copy to avoid numpy 1.8 warning
    W = gpu.to_gpu(W, allocator=_mempool.allocate)
    P = gpu.to_gpu(P, allocator=_mempool.allocate)
    X = gpu.to_gpu(X, allocator=_mempool.allocate)
    XXdiag = gpu.to_gpu(XXdiag, allocator=_mempool.allocate)
    I = la.eye(k, dtype=np.float32)

    init_rng(seed)
    t0 = time.time()
    for cur_iter in range(n_iter):
        H, tmp = calculate_H_gpu(X, W, P)
        if dropout_rate > 0:
            dropout(H, dropout_rate)
        Xtmp = X
        if input_dropout_rate > 0:
            Xtmp = X.copy()
            saltpepper_noise(Xtmp, input_dropout_rate)
        U = la.dot(Xtmp, H, "t", "n") / n
        S = la.dot(H, H, "t", "n") / n
        S += I
        S -= la.dot(tmp, W, "n", "t")
        Cii = la.dot(la.dot(W, S, "t") - 2*U, W)

        Sinv = la.inv(S, overwrite=True)
        dW = la.dot(Sinv, U, "n", "t") - W
        dP = XXdiag + la.diag(Cii) - P

        W += learnrateW * dW
        P += learnratePsi * dP

        P = gpu.maximum(P, minPsi)
        if cur_iter % 25 == 0:
            print "iter %3d (elapsed time: %5.2fs)" % (cur_iter, time.time() - t0)
    return W.get(), P.get()


def train_rfn_cpu(X, n_hidden, n_iter, learnrateW, learnratePsi, dropout_rate):
    n, m = X.shape
    k = n_hidden
    W = np.random.normal(scale=0.01, size=(k, m)).astype(np.float32)
    P = np.array([0.1] * m)
    H = np.zeros((k, n), dtype=np.float32)
    C = np.dot(X.T, X) / n

    t0 = time.time()
    for cur_iter in range(n_iter):
        I = np.eye(k, dtype=np.float32)
        tmp = I + np.dot(W * 1.0/P, W.T)
        tmp = np.linalg.inv(tmp)
        Wout = np.dot(tmp, W) * (1.0/P)
        H = np.dot(Wout, X.T)

        H = np.maximum(0, H)
        H /= (H.std(1) + 1e-9)[:, None]
        if dropout_rate > 0:
            H *= np.random.binomial(1, 1-dropout_rate, size=H.shape).astype(np.float32)

        U = np.dot(X.T, H.T) / n
        S = (np.dot(H, H.T) + tmp) / n

        dW = np.dot(np.linalg.inv(S), U.T) - W
        Cii = C - np.dot(-2*U + np.dot(W.T, S), W)
        dP = np.diag(Cii) - P

        W += learnrateW * dW
        P += learnratePsi * dP

        P = np.maximum(P, 0.1)

        if cur_iter % 25 == 0:
            print "iter %3d (elapsed time: %5.2fs)" % (cur_iter, time.time() - t0)
    return W, H, P
