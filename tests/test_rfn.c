#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../librfn.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GPU_ID 0
#define DEBUG_RESULT

// random in (0, 1]
static double rand_unif(void) {
    return (rand())/(RAND_MAX+1.0);
}

static double rand_max(int max) {
  return rand() % max;
}

int cmpfunc (const void * a, const void * b){
  return ( *(int*)a - *(int*)b );
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

sparse rowmajdense_to_csrsparse(const float *dense, int n, int m) {
    int nnz = 0;

    int* rowPointers = (int*) malloc((n + 1) * sizeof(int));
	  rowPointers[0] = 0;

	  for (int i = 0; i < n; i++) {
		    for (int j = 0; j < m; j++) {
            if (dense[i * m + j] != 0) {
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
			      if (dense[i * m + j] != 0) {
				        values[ind] = dense[i * m + j];
				        columns[ind] = j;
				        ind++;
			      }
		    }
	  }
	
	  sparse sp = { .vals = values, .indices = columns, .pointers = rowPointers, .n = n, .nnz = nnz};
	  return sp;
}

/*void print_csr(sparse csr, int n, int m) {
  int index = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (index < csr.pointers[i + 1]) {
        if (csr.indices[index] == j) {
          print( 
        }
    }
}*/

void free_sparse(sparse sp) {
    free(sp.vals);
    free(sp.indices);
    free(sp.pointers);
}

/*
// generates random samples from a 0/1 Gaussian via Box-Mueller
static double rand_normal(void) {
    return sqrt(-2.0*log(rand_unif())) * cos(2.0*M_PI*rand_unif());
}
*/


void printMat(float* x, int n, int m) {
	const char* format = "%1.3f ";
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j)
			printf(format, x[i + j * n]);
		printf("\n");
	}
	printf("\n");
}

void printfl(float*x, int n) {
	printMat(x, 1, n);
}

void printi(int* x, int n) {
	const char* format = "%d ";
	for (int i = 0; i < n; ++i) {
		printf(format, x[i]);
	}
	printf("\n");
}

double calculate_mean(double* array, int length) {
    double mean = 0;
    for (int i = 0; i < length; i++) {
        mean += array[i];
    }
    return mean / length;
}

double calculate_variance(double* array, double mean, int length) {
    double sum = 0;
    for (int i = 0; i < length; i++) {
        sum += (array[i] - mean) * (array[i] - mean);
    }
    return sum / length;
}


float* copy(const float *a, const int l) {
  float *val = malloc(l * sizeof(float));
  for (int i = 0; i < l; i++) {
    val[i] = a[i];
  }
  return val;
}


int main(int argc, char** argv) {
    srand(123);

    int n = 10000;
    int m = 784;
    int k = 5000;
    int n_iter = 3;
    int type = 1;
    float dropout = 0.95;
    int repeat_test = 3;

    if (argc > 1) {
        type = atoi(argv[1]);
    }

    if (argc > 2) {
        dropout = atof(argv[2]);
    }

    if (argc > 3) {
        repeat_test = atoi(argv[3]);
    }

    if (argc > 4) {
        n = atoi(argv[4]);
    }

    if (argc > 5) {
        m = atoi(argv[5]);
    }

    if (argc > 6) {
        k = atoi(argv[6]);
    }

    float* X = (float*) malloc(n * m * sizeof(float));
    for (int i = 0; i < n * m; ++i) {
        X[i] = rand_unif() < dropout ? 0 : (5.0f * rand_unif() - 0.5f);
    }

    sparse sp = rowmajdense_to_csrsparse(X, n, m);

    float* W = (float*) malloc(m * k * sizeof(float));
    float* P = (float*) malloc(m * sizeof(float));

    for (int i = 0; i < m * k; ++i) {
        W[i] = rand_unif() - 0.5;
    }
    for (int i = 0; i < m; ++i) {
        P[i] = 0.1;
    }
    double* times_spent = (double*) malloc(repeat_test * sizeof(double));
    clock_t begin, end;
    int retval;

    if (type & 1) {
    	printf("Testing GPU dense implementation.\n");
        for (int i = 0; i < repeat_test; i++) {
            float *W1 = copy(W, m * k);
            float *P1 = copy(P, m);
            srand(123);
            begin = clock();
            retval = train_rfn(X, W1, P1, n, m, k, n_iter, 2, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, GPU_ID);
            end = clock();
            times_spent[i] = (double)(end - begin) / CLOCKS_PER_SEC;
#ifdef DEBUG_RESULT
            printMat(W1, m, k);
#endif
        }
        double mean = calculate_mean(times_spent, repeat_test);
        double variance = calculate_variance(times_spent, mean, repeat_test);
    	  printf("Retval %d; Mean time spent: %3.4fs; Variance: %3.4f\n", retval, mean, variance);
    }
    if (type & (1 << 1)) {
        printf("Testing GPU sparse implementation.\n");
    	  for (int i = 0; i < repeat_test; i++) {
            float *W1 = copy(W, m * k);
            float *P1 = copy(P, m);
            srand(123);
            begin = clock();
            retval = train_rfn_sparse(sp.vals, sp.indices, sp.pointers, W1, P1, n, m, k, n_iter, 2, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, GPU_ID);
            end = clock();
            times_spent[i] = (double)(end - begin) / CLOCKS_PER_SEC;
#ifdef DEBUG_RESULT
            printMat(W1, m, k);
#endif
 
        }
        double mean = calculate_mean(times_spent, repeat_test);
        double variance = calculate_variance(times_spent, mean, repeat_test);
        printf("Retval %d; Mean time spent: %3.4fs; Variance: %3.4f\n", retval, mean, variance);
    }
    if (type & (1 << 2)) {
        printf("Testing CPU dense implementation.\n");
        begin = clock();
        retval = train_rfn(X, W, P, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, USE_CPU);
        end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("Retval %d; Time spent: %3.4fs\n", retval, time_spent);
    }
    if (type & (1 << 3)) {
        printf("Testing CPU sparse implementation.\n");
        begin = clock();
        retval = train_rfn_sparse(sp.vals, sp.indices, sp.pointers, W, P, n, m, k, n_iter, -1, 0.1, 0.1, 1e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1, 1, 32, USE_CPU);
        end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("Retval %d; Time spent: %3.4fs\n", retval, time_spent);
    }
    free(X);
    free(W);
    free(P);
    free_sparse(sp);
    free(times_spent);
    return 0;
}
