/*
Copyright © 2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/

#include "cpu_operations.h"
#include <cstdio>
#include <algorithm>

using std::printf;
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


void CPU_Operations::calculate_column_variance(const float* X, const unsigned nrows,
                                               const unsigned ncols, float* variances) {
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
    }
}


void CPU_Operations::invsqrt(float* s, const unsigned n) const {
    for (unsigned j = 0; j < n; ++j) {
        if (s[j] == 0)
            s[j] = 1.0f;
        else
            s[j] = 1.0 / sqrtf(s[j]);
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
