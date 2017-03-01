/*
Copyright © 2015-2017 Thomas Unterthiner
Additional Contributions by Thomas Adler, Balázs Bencze
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/

#include "librfn.h"
#include <stdlib.h>
#include <sys/time.h>
#include "cpu_operations.h"

#ifndef NOGPU
#include "gpu_operations.h"
#endif

#ifndef COMPILE_FOR_R
#include <stdio.h>
#include <assert.h>
#else
#include "use_R_impl.h"
#endif

float time_diff(struct timeval *t2, struct timeval *t1) {
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    return diff / 1000000.0f;
}

template <class OP>
int calculate_W_impl_invertMxM(OP& op, const float* W, const float* P, float* Wout,
                          const int k, const int m,
                          float* WWPchol, float* WWPinv) {
    op.gemm("n","t", m, m, k, 1.0f, W, m, W, m, 0.0f, WWPchol, m);
    op.axpy(m, 1.0f, P, 1, WWPchol, m+1);
    op.fill_eye(WWPinv, m);
    op.posv("u", m, m, WWPchol, m, WWPinv, m);
    op.gemm("t", "n", m, k, m, 1.0f, WWPinv, m, W, m, 0.0f, Wout, m);
    return 0;
}


//better option if m > k ( = W is tall), involves k*k inverse
template <class OP>
int calculate_W_impl_invertKxK(OP& op, const float* W, const float* Pinv, float* Wout,
                           const int k, const int m,
                           float* Wtmp, float* WPWchol, float* WPWinv) {
    op.dgmm("l", m, k, W, m, Pinv, 1, Wtmp, m);
    op.gemm("t", "n", k, k, m, 1.0f, W, m, Wtmp, m, 0.0f, WPWchol, k);
    op.axpy(k, 1.0f, op.ones, 1, WPWchol, k+1);
    op.fill_eye(WPWinv, k);
    op.posv("u", k, k, WPWchol, k, WPWinv, k);
    op.gemm("n", "t", m, k, k, 1.0f, Wtmp, m, WPWinv, k, 0.0f, Wout, m);
    return 0;
}


// if isMoreHiddensThanFeatures is true, we will calculate the m*m inverse, otherwise the k*k one
template <class OP, bool isMoreHiddensThanFeatures, typename XType, typename XTypeConst>
int train(XTypeConst X_host, float* W_host, float* P_host, const int n, const int m,
          const int k, const int n_iter, int batch_size, const float etaW, const float etaP,
          const float minP, const float h_threshold,
          const float dropout_rate, const float input_noise_rate,
          const float l2_weightdecay, const float l1_weightdecay, const float momentum,
          const int input_noise_type, const int activation_type, const int apply_scaling,
          const int applyNewtonUpdate, unsigned long seed, int gpu_id) {
    if (batch_size == 1) {
        printf ("batch_size == 1 not supported, switching to full batch mode");
    }

    OP op(n, m, k, seed, gpu_id);
    XType X = op.to_device(X_host, m*n*sizeof(float));
    float* W = op.to_device(W_host, k*m*sizeof(float));
    float* P = op.to_device(P_host, m*sizeof(float));
    if (batch_size < 2) // no mini-batches, one batch=full dataset
        batch_size = n;
    int n_batches = n / batch_size;
    float* XCov_diag = op.malloc(m*sizeof(float));

    float* H = op.malloc(k*batch_size*sizeof(float));
    float* Wout = op.malloc(k*m*sizeof(float));
    float* variance_H = op.malloc(k*sizeof(float));
    float* S = op.malloc(k*k*sizeof(float));
    float* Schol = op.malloc(k*k*sizeof(float));
    float* U = op.malloc(m*k*sizeof(float));
    float* Sinv = op.malloc(k*k*sizeof(float));
    float* dW = op.malloc(m*k*sizeof(float));
    float* C = op.malloc(m*m*sizeof(float));

    XType Xtmp = op.template init_invalid<XType>();
    if (input_noise_rate > 0.0f)
    {
       Xtmp = op.template malloc_matrix<XType>(batch_size, m);
    }

    // which matrices of the following we use depends on which inverse we use
    float* WWPchol = 0;
    float* WWPinv = 0;
    float* WPWchol = 0;
    float* WPWinv = 0;
    float* Wtmp = 0;
    if (isMoreHiddensThanFeatures) {
        WWPchol = op.malloc(m*m*sizeof(float));
        WWPinv = op.malloc(m*m*sizeof(float));
    } else {
        WPWchol = op.malloc(k*k*sizeof(float));
        WPWinv = op.malloc(k*k*sizeof(float));
        Wtmp = op.malloc(m*k*sizeof(float));
    }
    float* dP = op.malloc(m*sizeof(float));

    if (!dP) {  // We've run out of memory somewhere
        op.free(dP);
        op.free(C);
        op.free(dW);
        op.free(Sinv);
        op.free(U);
        op.free(Schol);
        op.free(S);
        op.free(variance_H);
        op.free(Wout);
        op.free(H);
        op.free(WWPinv);
        op.free(WWPchol);
        op.free(WPWchol);
        op.free(WPWinv);
        op.free(Wtmp);
        op.free(Xtmp);
        op.free(XCov_diag);
        return -1;
    }
    struct timeval t0, t1;
    gettimeofday(&t0, 0);

    if (n == batch_size) {
        op.calculate_column_variance(X, batch_size, m, XCov_diag, 1e-6);
        //op.printMatrixRM(XCov_diag, 1, m, "%1.2e ");
    }

    for (int cur_iter = 0; cur_iter < n_iter; ++cur_iter) {
        if (cur_iter % 1 == 0) {
            gettimeofday(&t1, 0);
            printf("epoch: %4d  (time: %6.2fs)\n", cur_iter, time_diff(&t1, &t0));
        }
        for (int cur_batch = 0; cur_batch < n_batches; ++cur_batch) {
            if (isMoreHiddensThanFeatures) {
                calculate_W_impl_invertMxM<OP>(op, W, P, Wout, k, m, WWPchol, WWPinv);
            } else {
                op.invert(P, m);  // TODO: something better than inverting P twice,
                /* how about inverting P once into distinct mem? */
                calculate_W_impl_invertKxK<OP>(op, W, P, Wout, k, m, Wtmp, WPWchol, WPWinv);
                op.invert(P, m);
            }

            XType Xnoise;
            if (input_noise_type && input_noise_rate > 0.0f) {
                op.memcpy_matrix(Xtmp, X, batch_size, m, cur_batch);
                switch(input_noise_type) {
                    case 1:  // dropout noise
                        op.dropout(Xtmp, batch_size*m, input_noise_rate);
                        break;
                    case 2: // salt&pepper noise
                        op.add_saltpepper_noise(Xtmp, batch_size*m, input_noise_rate);
                        break;
                    case 3: // gauss noise
                        op.add_gauss_noise(Xtmp, batch_size*m, input_noise_rate);
                        break;
                    default:
                        printf("invalid noise type");
                        assert(false);
                }
                Xnoise = Xtmp;
            } else {
                /* in case of sparse X, this is a copy operation! */
                Xnoise = op.get_batch(X, m, cur_batch, batch_size);
            }

            op.gemm("t", "n", k, batch_size, m, 1.0f, Wout, m, Xnoise, m, 0.0f, H, k);

            if (!(input_noise_type && input_noise_rate > 0.0f))
            {
               /* free matrix only if it's sparse */
               //op.free_sparse(Xnoise);
            }

            switch (activation_type) {
                case 1: op.maximum(H, h_threshold, batch_size*k); break;
                case 2: op.leaky_relu(H, h_threshold, batch_size*k); break;
                case 3: op.sigmoid(H, batch_size*k); break;
                case 4: op.tanh(H, batch_size*k); break;
                default:
                    printf("invalid activation type");
                    assert(false);
            }

            if (apply_scaling) {
                op.calculate_column_variance(H, batch_size, k, variance_H, 1e-6);
                op.invsqrt(variance_H, k);
                op.scale_columns(H, batch_size, k, variance_H);
            }
            if (dropout_rate > 0.0f) {
                op.dropout(H, batch_size*k, dropout_rate);
            }
            op.gemm("n", "t", k, k, batch_size, 1.0f/batch_size, H, k, H, k, 0.0f, S, k);
            if (isMoreHiddensThanFeatures) {
                op.gemm("t", "n", k, k, m, -1.0f, Wout, m, W, m, 1.0f, S, k);
                op.axpy(k, 1.0f, op.ones, 0, S, k+1);
            } else {
                op.axpy(k*k, 1.0f, WPWinv, 1, S, 1);
            }
            XType XBatch = op.get_batch(X, m, cur_batch, batch_size);
            op.gemm("n", "t", m, k, batch_size, 1.0f/batch_size, XBatch, m, H, k, 0.0f, U, m);

            if (applyNewtonUpdate) {
                op.axpy(k, 1e-10, op.ones, 0, S, k+1);
                op.memcpy(Schol, S, k*k*sizeof(float));
                op.fill_eye(Sinv, k);
                op.posv("u", k, k, Schol, k, Sinv, k);
                op.gemm("n", "n", m, k, k, 1.0f, U, m, Sinv, k, momentum, dW, m);
                op.axpy(m*k, -(1.0f+l2_weightdecay), W, 1, dW, 1);
            } else {
                op.gemm("n", "n", m, k, k, -1.0f, W, m, S, k, momentum, dW, m);
                op.axpy(m*k, 1.0f, U, 1, dW, 1);

                if (l2_weightdecay > 0.0f) {
                    op.axpy(m*k, -l2_weightdecay, W, 1, dW, 1);
                }
            }
            op.gemm("n", "n", m, k, k, 1.0f, W, m, S, k, -2.0f, U, m);
            op.gemm("n", "t", m, m, k, 1.0f, U, m, W, m, 0.0f, C, m);

            if (batch_size < n) {
                op.calculate_column_variance(XBatch, batch_size, m, dP, 1e-6);
            } else {
                op.memcpy(dP, XCov_diag, m*sizeof(float));
            }
            op.free_sparse(XBatch);

            op.axpy(m, 1.0f, C, m+1, dP, 1);
            op.axpy(m, -1.0f, P, 1, dP, 1);

            op.axpy(m, etaP/n_batches, dP, 1, P, 1);
            op.axpy(m*k, etaW/n_batches, dW, 1, W, 1);

            op.maximum(P, minP, m);

            if (l1_weightdecay > 0.0f) {
                op.soft_threshold(W, l1_weightdecay, m*k);
            }
        }
    }
    op.free(dP);
    op.free(C);
    op.free(dW);
    op.free(Sinv);
    op.free(U);
    op.free(Schol);
    op.free(S);
    op.free(H);
    op.free(variance_H);
    op.free(Wout);
    op.free(WWPinv);
    op.free(WWPchol);
    op.free(WPWchol);
    op.free(WPWinv);
    op.free(Wtmp);
    op.free(Xtmp);
    op.free(XCov_diag);
    op.free_devicememory(X);

    op.to_host(W, W_host, m*k*sizeof(float));
    op.to_host(P, P_host, m*sizeof(float));
    return 0;
}


template <class OP, typename XType, typename XTypeConst>
void calculate_W(XTypeConst X_host, const float* W_host, const float* P_host,
                 float* Wout_host, const int n, const int m, const int k,
                 const int activation_type, const int apply_scaling,
                 const float h_threshold, int gpu_id) {
    OP op(n, m, k, 0, gpu_id);
    float* P_copy = (float*) malloc(m*sizeof(float));
    memcpy(P_copy, P_host, m*sizeof(float)); // we might need to invert P
    float* Wout = op.to_device(Wout_host, k*m*sizeof(float));
    float* W = op.to_device(W_host, k*m*sizeof(float));
    float* P = op.to_device(P_copy, m*sizeof(float));
    XType  X = op.to_device(X_host, n*m*sizeof(float));
    float* H = op.malloc(n*k*sizeof(float));
    float* variance_H = op.malloc(k*sizeof(float));

    if (k > m) {
        float* WWPchol = op.malloc(m*m*sizeof(float));
        float* WWPinv = op.malloc(m*m*sizeof(float));
        calculate_W_impl_invertMxM<OP>(op, W, P, Wout, k, m, WWPchol, WWPinv);
        op.free(WWPchol);
        op.free(WWPinv);
    } else {
        op.invert(P, m);
        float* WPWchol = op.malloc(k*k*sizeof(float));
        float* WPWinv = op.malloc(k*k*sizeof(float));
        float* Wtmp = op.malloc(m*k*sizeof(float));
        calculate_W_impl_invertKxK<OP>(op, W, P, Wout, k, m, Wtmp, WPWchol, WPWinv);
        op.free(Wtmp);
        op.free(WPWinv);
        op.free(WPWchol);
        op.invert(P, m);
    }

    op.gemm("t", "n", k, n, m, 1.0f, Wout, m, X, m, 0.0f, H, k);

    switch (activation_type) {
        case 1: op.maximum(H, h_threshold, n*k); break;
        case 2: op.leaky_relu(H, h_threshold, n*k); break;
        case 3: op.sigmoid(H,  n*k); break;
        case 4: op.tanh(H,  n*k); break;
        default:
            printf("invalid noise type");
            assert(false);
    }

    if (apply_scaling){
        op.calculate_column_variance(H, n, k, variance_H, 1e-6);
        op.invsqrt(variance_H, k);
        op.scale_rows(Wout, k, m, variance_H);
    }

    op.free(variance_H);
    op.free(H);
    op.to_host(Wout, Wout_host, k*m*sizeof(float));
    op.free_devicememory(W);
    op.free_devicememory(P);
    op.free_devicememory(X);
    free(P_copy);
}


extern "C" {

int train_rfn(const float* X, float* W, float* P, const int n,
              const int m, const int k, const int n_iter, int batch_size,
              const float etaW, const float etaP, const float minP, const float h_threshold,
              const float dropout_rate, const float input_noise_rate,
              const float l2_weightdecay, const float l1_weightdecay,
              const float momentum,
              const int input_noise_type, const int activation_type, const int apply_scaling,
              const int applyNewtonUpdate, unsigned long seed, const int gpu_id) {
    if (gpu_id == USE_CPU) {
        if (k > m) {
            return train<CPU_Operations, true, float *, const float *>(X, W, P, n, m, k,
                        n_iter, batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                        l2_weightdecay, l1_weightdecay, momentum, input_noise_type, activation_type, apply_scaling, applyNewtonUpdate, seed, -1);
        } else {
            return train<CPU_Operations, false, float *, const float *>(X, W, P, n, m, k,
                        n_iter, batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                        l2_weightdecay, l1_weightdecay, momentum, input_noise_type, activation_type, apply_scaling, applyNewtonUpdate, seed, -1);
        }
    } else {
#ifndef NOGPU
        if (k > m) {
            return train<GPU_Operations, true, float *, const float *>(X, W, P, n, m, k,
                        n_iter, batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                        l2_weightdecay, l1_weightdecay, momentum, input_noise_type, activation_type, apply_scaling, applyNewtonUpdate, seed, gpu_id);
        } else {
            return train<GPU_Operations, false, float *, const float *>(X, W, P, n, m, k,
                        n_iter, batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                        l2_weightdecay, l1_weightdecay, momentum, input_noise_type, activation_type, apply_scaling, applyNewtonUpdate, seed, gpu_id);
        }
#else
        fprintf(stderr, "librfn was compiled without GPU support");
#endif
    }
}


int train_rfn_sparse(const float* Xvals, const int* Xcols, const int *Xrowptr,
              float* W, float* P, const int n, const int m,
              const int k, const int n_iter, int batch_size, const float etaW,
              const float etaP, const float minP, const float h_threshold,
              const float dropout_rate, const float input_noise_rate,
              const float l2_weightdecay, const float l1_weightdecay,
              const float momentum,
              const int input_noise_type, const int activation_type, const int apply_scaling,
              const int applyNewtonUpdate, unsigned long seed, const int gpu_id) {
    if (gpu_id == USE_CPU) {
        const CPU_Operations::SparseMatrix X = CPU_Operations::create_sparse_matrix(Xvals, Xcols, Xrowptr, n, m);
        int retval = 0;
        if (k > m) {
            retval =  train<CPU_Operations, true, CPU_Operations::SparseMatrix, const CPU_Operations::SparseMatrix>((CPU_Operations::SparseMatrix) X, W, P, n, m, k,
                        n_iter, batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                        l2_weightdecay, l1_weightdecay, momentum, input_noise_type, activation_type, apply_scaling, applyNewtonUpdate, seed, -1);
        } else {
            retval = train<CPU_Operations, false, CPU_Operations::SparseMatrix, const CPU_Operations::SparseMatrix>((CPU_Operations::SparseMatrix) X, W, P, n, m, k,
                        n_iter, batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                        l2_weightdecay, l1_weightdecay, momentum, input_noise_type, activation_type, apply_scaling, applyNewtonUpdate, seed, -1);
        }
        //destroy(X);
        return retval;
    } else {
#ifndef NOGPU
        const GPU_Operations::SparseMatrix X = GPU_Operations::create_sparse_matrix(Xvals, Xcols, Xrowptr, n, m);
        if (k > m) {
            return train<GPU_Operations, true, GPU_Operations::SparseMatrix*, const GPU_Operations::SparseMatrix*>(&X, W, P, n, m, k,
                        n_iter, batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                        l2_weightdecay, l1_weightdecay, momentum, input_noise_type, activation_type, apply_scaling, applyNewtonUpdate, seed, gpu_id);
        } else {
            return train<GPU_Operations, false, GPU_Operations::SparseMatrix*, const GPU_Operations::SparseMatrix*>(&X, W, P, n, m, k,
                        n_iter, batch_size, etaW, etaP, minP, h_threshold, dropout_rate, input_noise_rate,
                        l2_weightdecay, l1_weightdecay, momentum, input_noise_type, activation_type, apply_scaling, applyNewtonUpdate, seed, gpu_id);
        }
#else
        fprintf(stderr, "librfn was compiled without GPU support");
#endif
    }
}


void calculate_W(const float* X, const float* W, const float* P, float* Wout,
                 const int n, const int m, const int k, const int activation_type,
                 const int  apply_scaling, const float h_threshold, int gpu_id) {
    if (gpu_id == USE_CPU) {
        return calculate_W<GPU_Operations, float *, const float *>(X, W, P, Wout, n, m, k, activation_type, apply_scaling, h_threshold, gpu_id);
    } else {
#ifndef NOGPU
        return calculate_W<CPU_Operations, float *, const float *>(X, W, P, Wout, n, m, k, activation_type, apply_scaling, h_threshold, -1);
#else
        fprintf(stderr, "librfn was compiled without GPU support");
#endif
    }
}


void calculate_W_sparse(const float* Xvals, const int* Xcols, const int *Xrowptr,
                     const float* W, const float* P, float* Wout,
                     const int n, const int m, const int k, const int activation_type,
                     const int  apply_scaling, const float h_threshold, int gpu_id) {
    if (gpu_id == USE_CPU) {
        const CPU_Operations::SparseMatrix X = CPU_Operations::create_sparse_matrix(Xvals, Xcols, Xrowptr, n, m);
        calculate_W<CPU_Operations, CPU_Operations::SparseMatrix, const CPU_Operations::SparseMatrix>(X, W, P, Wout, n, m, k, activation_type, apply_scaling, h_threshold, -1);
        //destroy(X);
     } else {
#ifndef NOGPU
        const GPU_Operations::SparseMatrix X = GPU_Operations::create_sparse_matrix(Xvals, Xcols, Xrowptr, n, m);
        calculate_W<GPU_Operations, GPU_Operations::SparseMatrix*, const GPU_Operations::SparseMatrix *>(&X, W, P, Wout, n, m, k, activation_type, apply_scaling, h_threshold, gpu_id);
#else
        fprintf(stderr, "librfn was compiled without GPU support");
#endif
    }
}

}
