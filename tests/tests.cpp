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
    op.calculate_column_variance(X, nrows, ncols, var, 1e-6);
    float* res = (float*) malloc(ncols*sizeof(X[0]));
    op.copy_to_host(var, res, ncols*sizeof(var[0]));
    for (size_t i = 0; i < 3; ++i) {
        CHECK(abs(res[i] - expected[i]) < 1e-3);
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
    float x_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0};
    float e_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0};
    int n = sizeof(x_h) / sizeof(x_h[0]);
    for (int i = 0; i < n; ++i)
        e_h[i] = 1.0f / sqrt(x_h[i]);
    op.invsqrt(x_h, n);
    for (size_t i = 0; i < n; ++i) {
        CHECK(abs(x_h[i] - e_h[i]) < 1e-3);
    }
}


TEST_CASE( "invsqrt gpu", "[operations]" ) {
    GPU_Operations op(6, 6, 6, 0, -1);
    float x_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0};
    float e_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0};
    int n = sizeof(x_h) / sizeof(x_h[0]);
    for (int i = 0; i < n; ++i)
        e_h[i] = 1.0f / sqrt(x_h[i]);
    float* x_d = op.to_device(x_h, sizeof(x_h));
    op.invsqrt(x_d, n);
    float* res = (float*) malloc(n * sizeof(x_h[0]));
    op.copy_to_host(x_d, res, n * sizeof(x_h[0]));
    for (size_t i = 0; i < n; ++i) {
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
    cpu_op.calculate_column_variance(X_h, n, m, var_h, 1e-6);
    gpu_op.calculate_column_variance(X_d, n, m, var_d, 1e-6);
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


typedef struct sparse {
    float *vals;
    int *indices; // nnz of row or column
    int *pointers; // n + 1 column or row
    int n;
    int nnz;
} sparse;

// colmajor dense
sparse colmajdense_to_cscsparse(const float* dense, const int n, const int m, const int lda) {
    int nnz = 0;

    int* colPointers = (int*) malloc((m + 1) * sizeof(int));
	  colPointers[0] = 0;

	  for (int i = 0; i < m; i++) {
		    for (int j = 0; j < n; j++) {
			      if (dense[i * lda + j] != 0) {
				        nnz++;
            }
        }
        colPointers[i + 1] = nnz;
		}

	  float* values = (float*) malloc(nnz * sizeof(float));
	  int* rows = (int*) malloc(nnz * sizeof(int));
	  int ind = 0;
	  for (int i = 0; i < m; i++) {
		    for (int j = 0; j < n; j++) {
			      if (dense[i * lda + j] != 0) {
				        values[ind] = dense[i * lda + j];
				        rows[ind] = j;
				        ind++;
			      }
		    }
	  }
	
	  sparse sp = { .vals = values, .indices = rows, .pointers = colPointers, .n = m, .nnz = nnz};
	  return sp;
}

sparse colmajdense_to_csrsparse(const float *dense, int n, int m, int lda) {
    int nnz = 0;

    int* rowPointers = (int*) malloc((n + 1) * sizeof(int));
	  rowPointers[0] = 0;

	  for (int i = 0; i < n; i++) {
		    for (int j = 0; j < m; j++) {
			      if (dense[j * lda + i] != 0) {
				        nnz++;
            }
        }
        rowPointers[i + 1] = nnz;
		}

	  float* values = (float*) malloc(nnz * sizeof(float));
	  int* columns = (int*) malloc(nnz * sizeof(int));
	  int ind = 0;
	  for (int i = 0; i < n; i++) {
		    for (int j = 0; j < m; j++) {
			      if (dense[j * lda + i] != 0) {
				        values[ind] = dense[j * lda + i];
				        columns[ind] = j;
				        ind++;
			      }
		    }
	  }
	
	  sparse sp = { .vals = values, .indices = columns, .pointers = rowPointers, .n = n, .nnz = nnz};
	  return sp;
}

void free_sparse(sparse sp) {
    free(sp.vals);
    free(sp.indices);
    free(sp.pointers);
}

TEST_CASE( "gemm", "[operations]" ) {
    // matrix size 2 * 3
    // column major
    float a[] = {
        1.0, 3.0,
        2.0, 1.0,
        5.0, 1.0
    };
    // a transposed, 3 * 2
    float at[] = {
        1.0, 2.0, 5.0,
        3.0, 1.0, 1.0
    };
    // matrix of size 3 * 4
    float b[] = {
        4.0, 1.0, 2.0,
        4.0, 3.0, 2.0,
        4.0, 4.0, 1.0,
        7.0, 5.0, 1.0
    };
    // matrix of size 2 * 4
    float c[] = {
        1.0, 3.0,
        3.0, 2.0,
        8.0, 3.0,
        4.0, 3.0
    };
    // matrix of size 2 * 4
    float e[] = {
        35.0, 39.0,
        49.0, 40.0,
        58.0, 43.0,
        56.0, 63.0
    };
    float e2[] = {
        32, 32,
        40, 34,
        34, 34,
        44, 27
    };
    
    int m = 2;
    int n = 4;
    int k = 3;
    int lda = m;
    int ldb = k;
    int ldc = m;
    float alpha = 2.0;
    float beta = 3.0;
    float *cc = (float*) malloc(ldc * n * sizeof(float));
    SECTION( "cpu" ) {
        CPU_Operations op(6, 6, 6, 0, -1);
        memcpy(cc, c, ldc * n * sizeof(float));
        
        op.gemm("n", "n", m, n, k, alpha, a, lda, b, ldb, beta, cc, ldc);

        for (int i = 0; i < ldc * n; ++i) {
            CHECK(abs(cc[i] - e[i]) < 1e-3);
        }
    }
    SECTION( "gpu" ) {
        GPU_Operations op(6, 6, 6, 0, -1);

        float *ah = op.to_device(a, lda * k * sizeof(float));
        float *bh = op.to_device(b, ldb * n * sizeof(float));
        float *ch = op.to_device(c, ldc * n * sizeof(float));

        op.gemm("n", "n", m, n, k, alpha, ah, lda, bh, ldb, beta, ch, ldc);

        float *cd = (float*) malloc(ldc * n * sizeof(float));
        op.to_host(ch, cd, ldc * n * sizeof(float));

        for (int i = 0; i < ldc * n; ++i) {
            CHECK(abs(cd[i] - e[i]) < 1e-3);
        }
        op.free(ah);
        op.free(bh);
        free(cd);
    }
    SECTION( "cpu sparse" ) {
        /*CPU_Operations op(6, 6, 6, 0, -1);
        //csr asp = colmajdense_to_csrsparse(a, m, k, lda);
        csc asp = colmajdense_to_cscsparse(a, m, k, lda);
        //csc bsp = colmajdense_to_cscsparse(b, ldb, n);

        //CPU_Operations::SparseMatrix as = op.create_sparse_matrix(asp.vals, asp.cols, asp.rowPtrs, m, k); 
        CPU_Operations::SparseMatrix as = op.create_sparse_matrix(asp.vals, asp.rows, asp.colPtrs, m, k); 
        //CPU_Operations::SparseMatrix bs = op.create_sparse_matrix(bsp.vals, bsp.rows, bsp.colPtrs, ldb, n);
        
        op.printMatrixCM(as, m, k, 0);

        memcpy(cc, c, ldc * n * sizeof(float));
         
        op.gemm("n", "n", m, n, k, alpha, as, lda, b, ldb, beta, cc, ldc);
        for (int i = 0; i < ldc * n; ++i) {
            //CHECK(abs(cc[i] - e[i]) < 1e-3);
        }*/

        //memcpy(cc, c, ldc * n * sizeof(float));

        //op.gemm("n", "n", m, n, k, alpha, a, lda, bs, ldb, beta, cc, ldc);
        //for (int i = 0; i < ldc * n; ++i) {
        //  CHECK(abs(cc[i] - e[i]) < 1e-3);
        //}
        //CPU_Operations::free_sparse_matrix(as);
        //CPU_Operations::free_sparse_matrix(bs);
        //free_csr(asp);
        //free_sparse(bsp);
   }
   SECTION( "gpu sparse right" ) {
        GPU_Operations op(6, 6, 6, 0, -1);

        // Here we use CSC format, to implicitly transpose the B matrix.
        // This is undone in the gemm interface for sparse matrix
        sparse bsp = colmajdense_to_csrsparse(b, k, n, ldb);
        GPU_Operations::SparseMatrix bsh = op.create_sparse_matrix(bsp.vals, bsp.indices, bsp.pointers, k, n);
        GPU_Operations::SparseMatrix *bsd = op.to_device(&bsh, sizeof(int));

        float *atd = op.to_device(at, lda * k * sizeof(float));
        float *cd = op.to_device(c, ldc * n * sizeof(float));

        op.gemm("t", "n", m, n, k, 1.0, atd, k, bsd, ldb, 0.0, cd, ldc);
        
        float *ch = (float*) malloc(ldc * n * sizeof(float));
        op.to_host(cd, ch, ldc * n * sizeof(float));

        for (int i = 0; i < ldc * n; ++i) {
            CHECK(abs(ch[i] - e2[i]) < 1e-3);
        }
        
        // free asp, bsp
        op.free(atd);
        op.free_devicememory(bsd);
        free(ch);
        free_sparse(bsp);
    }
    free(cc);
}

