#include "catch.hpp"
#include "../cpu_operations.h"
#include "../gpu_operations.h"
#include <iostream>

#include <sys/time.h>
float time_diff(struct timeval *t2, struct timeval *t1) {
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    return diff / 1000000.0f;
}



using namespace std;

TEST_CASE( "to_host_and_to_device", "[gpu]" ) {
    GPU_Operations op(6, 6, 6, 0, -1);
    float X_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float* X_d = op.to_device(X_h, sizeof(X_h));

    float* X2_h = (float*) malloc(sizeof(X_h));
    op.copy_to_host(X_d, X2_h, sizeof(X_h));
    for (size_t i = 0; i < sizeof(X_h)/sizeof(X_h[0]); ++i) {
        CHECK(X_h[i] == X2_h[i]);
    }
    free(X2_h);
    op.free(X_d);
}


template <class OP>
float* test_variance(OP& op, float* X, unsigned nrows, unsigned ncols, float* expected) {
    float* var = (float*) op.malloc(ncols*sizeof(X[0]));
    op.calculate_column_variance(X, nrows, ncols, var);
    float* res = (float*) malloc(ncols*sizeof(X[0]));
    op.copy_to_host(var, res, ncols*sizeof(var[0]));
    for (size_t i = 0; i < 3; ++i) {
        CHECK(res[i] == expected[i]);
    }
    free(res);
    return var;
}


TEST_CASE( "Calculate Variance", "[operations]" ) {
    GPU_Operations gpu_op(512, 512, 512, 0, -1);
    CPU_Operations cpu_op(512, 512, 512, 0, -1);
    float X_h[] = {1.0, 2.0, 3.0,
                   4.0, 6.0, 10.0};
    float expected[] = {2.25, 4, 12.25};
    float* res_h = test_variance(cpu_op, X_h, 2, 3, expected);
    cpu_op.free(res_h);
    float* X_d = gpu_op.to_device(X_h, sizeof(X_h));
    float* res_d = test_variance(gpu_op, X_d, 2, 3, expected);
    gpu_op.free(res_d);
    gpu_op.free(X_d);
}


// the pointer-to-memberfunction thingy is pretty ugly :(
template <class OP>
float* test_scale(OP& op,
                  void (OP::*scalefunc)(float*, unsigned int, unsigned int, float*) const,
                  float* X, float* s, unsigned nrows, unsigned ncols, float* expected) {
    float* scale = op.to_device(s, ncols*sizeof(X[0]));
    (op.*scalefunc)(X, nrows, ncols, scale);
    float* res = (float*) malloc(ncols*nrows*sizeof(X[0]));
    op.copy_to_host(X, res, ncols*nrows*sizeof(X[0]));
    for (size_t i = 0; i < nrows*ncols; ++i) {
        CHECK(expected[i] == res[i]);
    }
    free(res);
    return 0;
}


TEST_CASE( "Scale columns CPU", "[operations]" ) {
    CPU_Operations op(6, 6, 6, 0, -1);
    float X_h[] = {1.0, 2.0, 3.0,
                   4.0, 6.0, 10.0};
    float s_h[] = {1.0, 2.0, 3.0};
    float Exp_h[] = {1.0,  4.0, 9.0,
                     4.0, 12.0, 30.0};
    test_scale(op, &CPU_Operations::scale_columns, X_h, s_h, 2, 3, Exp_h);
}


TEST_CASE( "Scale columns GPU", "[operations]" ) {
    GPU_Operations op(6, 6, 6, 0, -1);
    float X_h[] = {1.0, 2.0, 3.0,
                   4.0, 6.0, 10.0};
    float s_h[] = {1.0, 2.0, 3.0};
    float Exp_h[] = {1.0,  4.0, 9.0,
                     4.0, 12.0, 30.0};
    float* X_d = op.to_device(X_h, sizeof(X_h));
    test_scale(op, &GPU_Operations::scale_columns, X_d, s_h, 2, 3, Exp_h);
    op.free(X_d);
}


TEST_CASE( "Scale rows CPU", "[operations]" ) {
    CPU_Operations op(6, 6, 6, 0, -1);
    float X_h[] = {1.0, 2.0,  3.0, 4.0, 5.0,
                   4.0, 6.0, 10.0, 1.0, 1.5};
    float s_h[] = {2.0, 4.0};
    float Exp_h[] = { 2.0,  4.0,  6.0, 8.0, 10.0,
                     16.0, 24.0, 40.0, 4.0, 6.0};
    test_scale(op, &CPU_Operations::scale_rows, X_h, s_h, 2, 5, Exp_h);
}


TEST_CASE( "Scale rows GPU", "[operations]" ) {
    GPU_Operations op(6, 6, 6, 0, -1);
    float X_h[] = {1.0, 2.0,  3.0, 4.0, 5.0,
                   4.0, 6.0, 10.0, 1.0, 1.5};
    float s_h[] = {2.0, 4.0};
    float Exp_h[] = { 2.0,  4.0,  6.0, 8.0, 10.0,
                     16.0, 24.0, 40.0, 4.0, 6.0};
    float* X_d = op.to_device(X_h, sizeof(X_h));
    test_scale(op, &GPU_Operations::scale_rows, X_d, s_h, 2, 5, Exp_h);
    op.free(X_d);
}


TEST_CASE( "invsqrt cpu", "[operations]" ) {
    CPU_Operations op(6, 6, 6, 0, -1);
    float x_h[] = {0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0};
    float e_h[] = {1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0};
    int n = sizeof(x_h) / sizeof(x_h[0]);
    for (int i = 1; i < n; ++i)
        e_h[i] = 1.0f / sqrt(x_h[i]);
    op.invsqrt(x_h, n);
    for (size_t i = 0; i < 3; ++i) {
        CHECK(abs(x_h[i] - e_h[i]) < 1e-3);
    }
}


TEST_CASE( "invsqrt gpu", "[operations]" ) {
    GPU_Operations op(6, 6, 6, 0, -1);
    float x_h[] = {0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0};
    float e_h[] = {1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0};
    int n = sizeof(x_h) / sizeof(x_h[0]);
    for (int i = 1; i < n; ++i)
        e_h[i] = 1.0f / sqrt(x_h[i]);
    float* x_d = op.to_device(x_h, sizeof(x_h));
    op.invsqrt(x_d, n);
    float* res = (float*) malloc(n*sizeof(x_h[0]));
    op.copy_to_host(x_d, res, n*sizeof(x_h[0]));
    for (size_t i = 0; i < 3; ++i) {
        CHECK(abs(res[i] - e_h[i]) < 1e-3);
    }
    op.free(x_d);
}


TEST_CASE( "filleye cpu", "[operations]" ) {
    unsigned n = 10;
    CPU_Operations op(n, n, n, 0, -1);
    float* x = op.malloc(n*n*sizeof(float));
    op.fill_eye(x, 10);
    double s = 0.0;
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            if (i == j) {
                CHECK(x[i*n+j] == 1.0);
            } else {
                s += abs(x[i*n+j]);
            }
        }
    }
    CHECK(s == 0.0);
    op.free(x);
}


TEST_CASE( "filleye gpu", "[operations]" ) {
    unsigned n = 10;
    CPU_Operations cpu_op(n, n, n, 0, -1);
    GPU_Operations op(n, n, n, 0, -1);
    float* x_d = op.malloc(n*n*sizeof(float));
    op.fill_eye(x_d, 10);
    float *x = cpu_op.malloc(n*n*sizeof(float));
    op.copy_to_host(x_d, x, n*n*sizeof(float));
    double s = 0.0;
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            if (i == j) {
                CHECK(x[i*n+j] == 1.0);
            } else {
                s += abs(x[i*n+j]);
            }
        }
    }
    CHECK(s == 0.0);
    op.free(x_d);
}


TEST_CASE( "Variance of CPU/GPU on large matrices", "[cpu_vs_gpu]" ) {
    unsigned n = 428;
    unsigned m = 554;
    CPU_Operations cpu_op(m, n, m, 0, -1);
    GPU_Operations gpu_op(m, n, m, 0, -1);

    float* X_h = cpu_op.malloc(n*m*sizeof(float));
    for (unsigned i = 0; i < n*m; ++i) {
        X_h[i] = 10*((rand()+1.0)/(RAND_MAX+1.0)) - 5.0;
    }
    float *X_d = gpu_op.to_device(X_h, n*m*sizeof(float));

    float* var_h = cpu_op.malloc(m*sizeof(float));
    float* var_d = gpu_op.malloc(m*sizeof(float));
    cpu_op.calculate_column_variance(X_h, n, m, var_h);
    gpu_op.calculate_column_variance(X_d, n, m, var_d);
    float* var_gpu_h = cpu_op.malloc(m*sizeof(float));
    gpu_op.to_host(var_d, var_gpu_h, m*sizeof(float));

    for (unsigned i = 0; i < m; ++i)
        CHECK(abs(var_h[i] - var_gpu_h[i]) < 1e-3);
    cpu_op.free(var_h);
    cpu_op.free(var_gpu_h);
}



TEST_CASE( "dgmm CPU/GPU", "[operations]" ) {
    unsigned n = 10;
    unsigned k = 10;
    unsigned m = 12;
    CPU_Operations cpu_op(m, n, k, 0, -1);
    GPU_Operations gpu_op(m, n, k, 0, -1);
    float* xh = cpu_op.malloc(m*k*sizeof(float));
    float* ah = cpu_op.malloc(m*sizeof(float));
    float* ch = cpu_op.malloc(m*k*sizeof(float));
    for (int i = 0; i < m*n; ++i)
        xh[i] = 10* (rand() / RAND_MAX);
    for (int i = 0; i < n; ++i)
        ah[i] = 50* (rand() / RAND_MAX);
    cpu_op.dgmm("l", m, k, xh, m, ah, 1, ch, m);

    float* xd = gpu_op.to_device(xh, m*k*sizeof(float));
    float* ad = gpu_op.to_device(ah, m*sizeof(float));
    float* cd = gpu_op.to_device(ch, m*k*sizeof(float));
    gpu_op.dgmm("l", m, k, xd, m, ad, 1, cd, m);

    float* dh = cpu_op.malloc(m*k*sizeof(float));
    gpu_op.copy_to_host(cd, dh, m*k*sizeof(float));
    for (unsigned i = 0; i < m*k; ++i) {
        CHECK(ch[i] == dh[i]);
    }
}
