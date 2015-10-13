# librfn: Rectified Factor Networks

Rectified Factor Networks (RFNs) are an unsupervised technique that learns a non-linear, high-dimensional representation of its input. The underlying algorithm has been published in

*Rectified Factor Networks*, Djork-Arné Clevert, Andreas Mayr, Thomas Unterthiner, Sepp Hochreiter, NIPS 2015.

librfn is implemented in C++ and can be easily integrated in existing code bases. It also contains a high-level Python wrapper for ease of use. The library can run in either CPU or GPU mode. For larger models the GPU mode offers large speedups and is the recommended mode.


# Installation

1. (optional) Adjust the Makefile to your needs
2. Type `make` to start the building process
3. To use the python wrapper, just copy `rfn.py` and `librfn.so` into your  working directory.


# Requirements

To run the GPU code, you require a CUDA 7.5 compatible GPU. While in theory CUDA 7.0 is also supported, it contains a bug that results in a memory leak when running librfn (and your program is likely to crash with an out-of-memory error).

If you do not have access to a GPU, you can disable GPU support by setting `USEGPU = no` in the Makefile.

Note that librfn makes heavy use of BLAS and LAPACK, so make sure to link it to a high-quality implementation to get optimal speed (e.g. OpenBLAS or MKL) by modifying the Makefile.


# Usage

The following code trains a RFN on MNIST and plots the resulting filters::

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    X = mnist['data'] / 255.0

    from rfn import *
    W, P = train_rfn(X, 128, 500, 0.1, 0.1, 1e-1, 0.0, gpu_id=0)

    # plot weights
    fig, ax = plt.subplots(5, 5, figsize=(8, 8))
    for i, a in enumerate(ax.flat):
        a.pcolorfast(W[i].reshape(28, 28), cmap=plt.cm.Greys_r)
        a.set_ylim(28, 0)
        a.grid("off")
        a.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig

    # calculate hidden units and reconstructions
    H = np.maximum(0, np.dot(Wout, X.T))
    R = np.dot(H.T, W)

    # plot reconstructions
    np.random.shuffle(R)  # shuffle samples, otherwhise we only plot 0s
    fig, ax = plt.subplots(5, 5, figsize=(8, 8))
    for i, a in enumerate(ax.flat):
        a.pcolorfast(R[i].reshape(28, 28), cmap=plt.cm.Greys_r)
        a.set_ylim(28, 0)
        a.grid("off")
        a.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig


# Implementation Note

The RFN algorithm is based on the EM algorithm. Within the E-step, the published algorithm includes a projection procedure that can be implemented in several ways (see the RFN paper's supplemental section 9). To make sure no optimzation constraints are violated during this projection, the original publication tries the simplest method first, but backs out to more and more complicated updates if easier method fail (suppl. section 9.5.3).
In contrast, librfn always uses the simplest/fastest projection method. This is a simplification/approximation of the original algorithm that nevertheless works very well in practice.


# License
librfn is licensed under the [General Public License (GPL) Version 2 or higher](http://www.gnu.org/licenses/gpl-2.0.html) See ``License.txt`` for details.
