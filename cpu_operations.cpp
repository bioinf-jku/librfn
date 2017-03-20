/*
Copyright © 2015-2017 Thomas Unterthiner
Additional Contributions by Thomas Adler, Balázs Bencze
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/

#include "cpu_operations.h"
#include <algorithm>


/* This is the interface for RFN's sparse matrix operations.
 * If you want to use the generic implementation, compile nist_spblas.cc,
 * If you want to use the MKL, compile mkl_sparse_impl.cpp and link to MKL. */

CPU_Operations::SparseMatrix create(int row, int col); /* empty */
CPU_Operations::SparseMatrix suscr_csr(int m, int n, float *val, int *col, int *ptr); /* from csr */
void destroy(CPU_Operations::SparseMatrix A);

/* select row subset */
CPU_Operations::SparseMatrix srowsubset(CPU_Operations::SparseMatrix A, int first_row, int nrow); /* allocates new matrix */

/* column means and variances */
void scolmeans(CPU_Operations::SparseMatrix A, float *means);
void scolvars(CPU_Operations::SparseMatrix A, float *vars);

/* scale rows/cols */
void sscalecols(CPU_Operations::SparseMatrix A, float *s);
void sscalerows(CPU_Operations::SparseMatrix A, float *s);

/* set element (set to zero will delete entry) */
void ssetelement( CPU_Operations::SparseMatrix A, int row, int col, float val );
void ssetelement( CPU_Operations::SparseMatrix A, int idx, float val );

/* get element reference */
float &sgetelement( CPU_Operations::SparseMatrix A, int row, int col);
float &sgetelement( CPU_Operations::SparseMatrix A, int idx );

/* get element pointer */
float *sgetelementp( CPU_Operations::SparseMatrix A, int row, int col );
float *sgetelementp( CPU_Operations::SparseMatrix A, int idx );

/* sgemm routines with sparse matrix being lhs (A) or rhs (B) of the product */
void susgemm(char sidea, char transa, char transb, int nohs, const float &alpha, CPU_Operations::SparseMatrix A,
   const float *B, int ldB, const float &beta, float *C, int ldC);

/* checks whether A is a valid handle */
bool handle_valid(CPU_Operations::SparseMatrix A);

/* debug */
namespace NIST_SPBLAS
{void print(int A);}


using std::max;

//float* CPU_Operations::ones = 0;

CPU_Operations::CPU_Operations(const int m, const int n, const int k,
                          unsigned long seed, int gpu_id) {
    srand(seed);
    int maxsize = max(max(n, m), k);
    ones = malloc(maxsize*sizeof(float));
    for (int i = 0; i < maxsize; ++i)
        ones[i] = 1.0f;

    var_tmp = malloc(maxsize*sizeof(float));
}


CPU_Operations::~CPU_Operations() {
    free(ones);
    free(var_tmp);
}

CPU_Operations::SparseMatrix CPU_Operations::create_sparse_matrix(const float* Xvals, const int* Xcols, const int *Xrowptr, int n, int m){
    return suscr_csr(n, m, (float*) Xvals, (int*) Xcols, (int*) Xrowptr);
}

void CPU_Operations::free_sparse_matrix(CPU_Operations::SparseMatrix x) {
    if (handle_valid(x))
        destroy(x);
}

CPU_Operations::SparseMatrix CPU_Operations::get_batch(SparseMatrix X, int ldx, int batch_num, int batch_size) {
    return srowsubset(X, batch_num * batch_size, batch_size);
}

void CPU_Operations::scale_rows(SparseMatrix X, const unsigned nrows, const unsigned ncols, float* s) const {
        sscalerows(X, s);
}


static void colmeans(const float* X, float* means, const int nrows, const int ncols) {
    memset(means, 0, ncols*sizeof(float));
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            means[j] += X[i*ncols+j];
        }
    }
    for (int j = 0; j < ncols; ++j)
        means[j] /= nrows;
}


void CPU_Operations::dgmm(const char* mode, const int m, const int n, const float* A,
                  int lda, const float* x, int incx, float* C, int ldc) const {
    if (mode[0] == 'l' || mode[0] == 'L') {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j)
                C[i*ldc+j] = A[i*lda+j] * x[j];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j)
                C[i*ldc+j] = A[i*lda+j] * x[i];
        }
    }
}


void CPU_Operations::gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
        const SparseMatrix a, const int lda, const float *b, const int ldb, const float beta, float *c,
        const int ldc) const {
    /* The gemm interface is understood as a column-major routine. The sparse implementation,
     * however, is row-major, so we need to compute B^T * A^T = C^T instead of A * B = C. The
     * transposition is implicitly performed by A, B and C being column-major. */
    susgemm('r', transa[0], transb[0], n, alpha, a, b, ldb, beta, c, ldc);
}

void CPU_Operations::gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
        const float *a, const int lda, const SparseMatrix b, const int ldb, const float beta, float *c,
        const int ldc) const {
    susgemm('l', transb[0], transa[0], m, alpha, b, a, lda, beta, c, ldc);
}

CPU_Operations::SparseMatrix CPU_Operations::memcpy_matrix(SparseMatrix &dest, SparseMatrix src, int nrows_to_copy, int src_ncol, int first_row = 0) const {
    free(dest);
    return dest = srowsubset(src, first_row, nrows_to_copy);
}

void CPU_Operations::free(SparseMatrix a) const {
    if (handle_valid(a))
        destroy(a);
}

CPU_Operations::SparseMatrix CPU_Operations::malloc_matrix(int rows, int cols, SparseMatrix dummy) {
    return create(rows, cols);
}

void CPU_Operations::calculate_column_variance(SparseMatrix X, const unsigned nrows, const unsigned ncols, float* variances, float eps) {
    memset(variances, 0, ncols * sizeof(float));
    scolvars(X, variances);

    // for numerical stability of the algorithm
    for (unsigned i = 0; i < ncols; ++i)
        variances[i] += eps;
}

void CPU_Operations::scale_columns(SparseMatrix X, const unsigned nrows, const unsigned ncols, float* s) const {
    sscalecols(X, s);
}

void CPU_Operations::dropout(SparseMatrix X, const unsigned size, const float dropout_rate) const {
    assert(0.0f <= dropout_rate && dropout_rate <= 1.0f);
    for (unsigned i = 0; i < size; ++i)
        /* TODO: write a routine sgetlement that leaves X const */
        if (rand_unif() < dropout_rate) {
            float *v = sgetelementp(X, i);

            if (v != NULL)
                *v = 0.f;
        }
}

void CPU_Operations::add_saltpepper_noise(SparseMatrix X, const unsigned size, const float noise_rate) const {
    assert(0.0f <= noise_rate && noise_rate <= 1.0f);
    for (unsigned i = 0; i < size; ++i) {
        if (rand_unif() < noise_rate) {
            float *v = sgetelementp(X, i);

            if (v != NULL)
                *v = (rand_unif() < 0.5 ? 0.0f : 1.0f);
        }
    }
}

/* gauss noise makes no sense on sparse matrices */
void CPU_Operations::add_gauss_noise(SparseMatrix X, const unsigned size, const float noise_rate) const {
    assert(0.0 <= noise_rate);
    for (unsigned i = 0; i < size; ++i) {
        float *v = sgetelementp(X, i);

        if (v != NULL)
            *v += rand_normal() * noise_rate;
    }
}


void CPU_Operations::calculate_column_variance(const float* X, const unsigned nrows,
                                               const unsigned ncols, float* variances, float eps) {
    colmeans(X, var_tmp, nrows, ncols);
    memset(variances, 0, ncols*sizeof(float));
    for (unsigned i = 0; i < nrows; ++i) {
        for (unsigned j = 0; j < ncols; ++j) {
            const float x = X[i*ncols+j] - var_tmp[j];
            variances[j] += x*x;
        }
    }

    for (unsigned j = 0; j < ncols; ++j) {
        variances[j] /= nrows;
        variances[j] += eps; // for numerical stability
    }
}


void CPU_Operations::invsqrt(float* s, const unsigned n) const {
    for (unsigned j = 0; j < n; ++j) {
        if (s[j] == 0)
            s[j] = 1.0f;
        else
            s[j] = 1.0 / sqrtf(s[j] + 1e-8);
    }
}

void CPU_Operations::scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const {
    for (unsigned i = 0; i < nrows; ++i) {
        for (unsigned j = 0; j < ncols; ++j) {
            X[i*ncols+j] *= s[j];
        }
    }
}

void CPU_Operations::scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const {
    for (unsigned i = 0; i < nrows; ++i) {
        for (unsigned j = 0; j < ncols; ++j) {
            X[i*ncols+j] *= s[i];
        }
    }
}


/// Prints a column major matrix.
void CPU_Operations::printMatrixCM(const float* a, int n, int m, const char* fmt) {
    const char* format = fmt == 0 ? "%1.3f " : fmt;
    for (int i = 0; i < n; ++i) {
        for (int j =0 ; j < m; ++j)
            printf(format, a[i + j*n]);
        printf("\n");
    }
    printf("\n");
}


/// Prints a row-major matrix
void CPU_Operations::printMatrixRM(const float* a, int n, int m, const char* fmt) {
    const char* format = fmt == 0 ? "%1.3f " : fmt;
    for (int i = 0; i < n; ++i) {
        for (int j =0 ; j < m; ++j)
            printf(format, a[i*m + j]);
        printf("\n");
    }
}

void CPU_Operations::printMatrixCM(const SparseMatrix a, int n, int m, const char *fmt) {
    NIST_SPBLAS::print(a);
}

void CPU_Operations::printMatrixRM(const SparseMatrix a, int n, int m, const char *fmt) {
    NIST_SPBLAS::print(a);
}
