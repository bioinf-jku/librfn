/*
Copyright © 2015-2017 Thomas Unterthiner
Additional Contributions by Thomas Adler, Balázs Bencze
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <typeinfo> /* for typeid */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
extern void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha,
        const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c, const int *ldc);

extern void ssymm_(const char *side, const char *uplo, const int *m, const int *n, const float *alpha, const float *a,
        const int *lda, const float *b, const int *ldb, const float *beta, float *c, const int *ldc);

extern void saxpy_(const int *n, const float *alpha, const float *dx, const int *incx, float *dy, const int *incy);
extern int spotrf_(const char *uplo, int *n, float *a, int * lda, int *info);
extern int spotrs_(const char *uplo, int *n, int *nrhs, float * a, int *lda, float *b, int *ldb, int *info);
extern int sposv_(const char *uplo, int *n, int *nrhs, float * a, int *lda, float *b, int *ldb, int *info);
extern int spotri_(const char *uplo, int *n, float *a, int *lda, int *info);
}

using std::cos;
using std::log;
using std::sqrt;

#ifdef COMPILE_FOR_R
#include "use_R_impl.h"
#else
#include <cstdio>
#include <cassert>

using std::rand;
using std::srand;

// random in (0, 1]
inline double rand_unif(void) {
    return (rand() + 1.0) / (RAND_MAX + 1.0);
}

// generates random samples from a 0/1 Gaussian via Box-Mueller
inline double rand_normal(void) {
    return sqrt(-2.0 * log(rand_unif())) * cos(2.0 * M_PI * rand_unif());
}
#endif

inline double rand_exp(double lambda) /* inversion sampling */
{
    return -log(1 - rand_unif()) / lambda;
}

class CPU_Operations {
    float* var_tmp;

public:

float* ones;

typedef int SparseMatrix;

static SparseMatrix create_sparse_matrix(const float* Xvals, const int* Xcols, const int *Xrowptr, int n, int m);
static void free_sparse_matrix(SparseMatrix);


template<typename T>
T init_invalid(void) {
    return (typeid(T) == typeid(SparseMatrix) ? (T) -1 : (T) 0);
}

CPU_Operations(const int m, const int n, const int k, unsigned long seed, int gpu_id);
~CPU_Operations();

float* to_device(const float* src, const int size) const {
    return (float*) src;
}

SparseMatrix to_device(SparseMatrix src, const int size) const {
    return src;
}

float* to_host(const float* src, float* dest, const int size) const {
    return dest;
}

float* copy_to_host(const float* src, float* dst, size_t size) const {
    memcpy(dst, src, size);
    return dst;
}

float* get_batch(const float* X, int ncol, int batch_num, int batch_size) {
    /* return pointer */
    return (float*) &X[batch_num * batch_size * ncol];
}

SparseMatrix get_batch(SparseMatrix X, int ldx, int batch_num, int batch_size);

void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
        const float *a, const int lda, const float *b, const int ldb, const float beta, float *c,
        const int ldc) const {
    sgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}


void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
        const SparseMatrix a, const int lda, const float *b, const int ldb, const float beta, float *c,
        const int ldc) const;

void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
        const float *a, const int lda, const SparseMatrix b, const int ldb, const float beta, float *c,
        const int ldc) const;

void dgmm(const char* mode, const int m, const int n, const float* A, int lda, const float* x, int incx, float* C,
        int ldc) const;

void symm(const char *side, const char *uplo, const int m, const int n, const float alpha, const float *a,
        const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc) const {
    ssymm_(side, uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

void axpy(const int n, const float alpha, const float *dx, const int incx, float *dy, const int incy) const {
    saxpy_(&n, &alpha, dx, &incx, dy, &incy);
}

int posv(const char* uplo, int n, int nrhs, float* a, int lda, float* b, int ldb) const {
    int info;
    int retval = sposv_(uplo, &n, &nrhs, a, &lda, b, &ldb, &info);

    if (info != 0)
        printf("info: %d\n", info);

    assert(!info);

    return retval;
}

int potrf(const char *uplo, int n, float* a, int lda) const {
    int info;
    int retval = spotrf_(uplo, &n, a, &lda, &info);
    assert(!info);
    return retval;
}

int potrs(const char *uplo, int n, int nrhs, float* a, int lda, float *b, int ldb, int *info) const {
    return spotrs_(uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

int potri(const char *uplo, int n, float *a, int lda) const {
    int info;
    int retval = spotri_(uplo, &n, a, &lda, &info);
    assert(!info);
    return retval;
}

void* memset(void* dest, int ch, size_t count) const {
    return std::memset(dest, ch, count);
}

float* memcpy(void* dest, const void *src, size_t count) const {
    return (float*) std::memcpy(dest, src, count);
}

float *memcpy_matrix(float *dest, float *src, int nrows_to_copy, int src_ncol, int first_row = 0) const {
    return memcpy(dest, &src[first_row * src_ncol], nrows_to_copy * src_ncol * sizeof(float));
}

SparseMatrix memcpy_matrix(SparseMatrix &dest, SparseMatrix src, int nrows_to_copy, int src_ncol, int first_row) const;

void free(void* ptr) const {
    if (ptr != 0)
        std::free(ptr);
}

void free(SparseMatrix a) const;

void free_batch(void *ptr) {
}

void free_batch(SparseMatrix a) {
    free(a);
}

void free_devicememory(void* ptr) const {
    ;
}

void free_devicememory(SparseMatrix X) const {
}

template<typename T>
T malloc_matrix(int rows, int cols) {
    return malloc_matrix(rows, cols, init_invalid<T>());
}

SparseMatrix malloc_matrix(int rows, int cols, SparseMatrix dummy);

float *malloc_matrix(int rows, int cols, float *dummy) {
    return malloc(rows * cols * sizeof(float));
}

float* malloc(size_t size) const {
    return (float*) std::malloc(size);
}

void maximum(float* x, const float value, const int size) const {
    for (int i = 0; i < size; ++i)
        x[i] = fmaxf(x[i], value);
}

void leaky_relu(float* x, const float value, const int size) {
    for (int i = 0; i < size; ++i)
        x[i] = (x[i] < 0.0f) ? x[i] * value : x[i];
}

void sigmoid(float* x, const int size) const {
    for (int i = 0; i < size; ++i) {
        x[i] = 1 / (1 + expf(-x[i]));
    }
}

void tanh(float* x, const int size) const {
    for (int i = 0; i < size; ++i) {
        x[i] = tanhf(x[i]);
    }
}

void fill_eye(float* a, int n) const {
    memset(a, 0, n * n * sizeof(float));
    for (int i = 0; i < n; ++i)
        a[i * n + i] = 1.0f;
}

void fill(float* X, const int size, const float value) const {
    for (int i = 0; i < size; ++i) {
        X[i] = value;
    }
}

void calculate_column_variance(const float* X, const unsigned nrows, const unsigned ncols, float* variances, float eps);
void calculate_column_variance(SparseMatrix X, const unsigned nrows, const unsigned ncols, float* variances, float eps);

void invsqrt(float* s, const unsigned n) const;

void scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
void scale_columns(SparseMatrix X, const unsigned nrows, const unsigned ncols, float* s) const;

void scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
void scale_rows(SparseMatrix X, const unsigned nrows, const unsigned ncols, float* s) const;

void dropout(float* X, const unsigned size, const float dropout_rate) const {
    assert(0.0f <= dropout_rate && dropout_rate <= 1.0f);
    for (unsigned i = 0; i < size; ++i)
        X[i] = rand_unif() < dropout_rate ? 0.0f : X[i];
}

void dropout(SparseMatrix X, const unsigned size, const float dropout_rate) const;

void add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const {
    assert(0.0f <= noise_rate && noise_rate <= 1.0f);
    for (unsigned i = 0; i < size; ++i) {
        if (rand_unif() < noise_rate) {
            X[i] = rand_unif() < 0.5 ? 0.0f : 1.0f;
        }
    }
}

void add_saltpepper_noise(SparseMatrix X, const unsigned size, const float noise_rate) const;

void add_gauss_noise(float* X, const unsigned size, const float noise_rate) const {
    assert(0.0 <= noise_rate);
    for (unsigned i = 0; i < size; ++i)
        X[i] += rand_normal() * noise_rate;
}

/* gauss noise makes no sense on sparse matrices */
void add_gauss_noise(SparseMatrix X, const unsigned size, const float noise_rate) const;

void invert(float* X, const unsigned size) const {
    for (unsigned i = 0; i < size; ++i)
        X[i] = 1.0f / X[i];
}

void soft_threshold(float* x, const float alpha, const int size) const {
    float f;
    for (int i = 0; i < size; ++i) {
        f = x[i];
        x[i] = f > 0 ? fmaxf(0., f - alpha) : fminf(0., f + alpha);
    }
}

// Useful for debugging
static void printMatrixCM(const float* a, int n, int m, const char* fmt);
static void printMatrixCM(const SparseMatrix a, int n, int m, const char *fmt);

static void printMatrixRM(const float* a, int n, int m, const char* fmt);
static void printMatrixRM(const SparseMatrix a, int n, int m, const char *fmt);

void printMatrixSP(const SparseMatrix& a, const char* fmt) const {
    // TODO
}

void prints(const float* f, unsigned l) const {}

void printsu(const int* f, unsigned l) const {}

void printm(const char* name, const SparseMatrix a, int n, int m) const {
    printf("%s\n", name);
    printMatrixCM(a, n, m, 0);
}

void printm(const char* name, const float* a, int n, int m) const {
    printf("%s\n", name);
    printMatrixCM(a, n, m, 0);
}
};
