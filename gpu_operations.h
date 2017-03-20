/*
Copyright © 2015-2017 Thomas Unterthiner
Additional Contributions by Thomas Adler, Balázs Bencze
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusolverDn.h>
#include <cstring>
#include <ctype.h>
#include <map>
#include <cusparse_v2.h>
#include <typeinfo> /* for typeid */

#ifndef COMPILE_FOR_R
#include <stdio.h>
#include <assert.h>
#else
#include "use_R_impl.h"
#endif



// This code to print a stack trace will only work on Linux
#ifndef NDEBUG
#include <execinfo.h>
static void print_stacktrace() {
    const int MAX_SIZE = 15;
    void *array[MAX_SIZE];
    int size = backtrace(array, MAX_SIZE);
    char **strings = 0;
    strings = backtrace_symbols (array, size);
    printf("Obtained %d stack frames:\n", size);
    for (int i = 0; i < size; ++i)
        printf ("%s\n", strings[i]);
    free (strings);
}
#else
static void print_stacktrace() {
}
#endif


using std::fprintf;

inline cublasFillMode_t uplo_to_cublas(const char* uplo) {
    return tolower(uplo[0]) == 'l' ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
}

inline cusparseOperation_t op_to_cusparse(const char* op) {
    return tolower(op[0]) == 't' ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
}

static const char* cusparseErrorString(cusparseStatus_t error) {
    switch (error) {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
        default: return "<unknown>";
    }
}

static const char* cublasErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
        default: return "<unknown>";
    }
}

#ifndef DNDEBUG

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        print_stacktrace();
        if (abort)
            exit(code);
    }
}

#define CUBLAS_CALL(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line) {
//printf("%d (%s:%d)\n", code, file, line);
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Error: %s %s:%d\n", cublasErrorString(code), file, line);
        print_stacktrace();
        exit(code);
    }
}

#define CUSPARSE_CALL(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line) {
    // printf("%d (%s:%d)\n", code, file, line);
    if (code != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE Error: %s %s:%d\n", cusparseErrorString(code), file, line);
        print_stacktrace();
        exit(code);
    }
}

static const char* cusolverErrorString(cusolverStatus_t error) {
    switch (error) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default: return "<unknown>";
    }
}

#define CUSOLVER_CALL(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
    //printf("%d (%s:%d)\n", code, file, line);
    if (code != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Error: %s %s:%d\n", cusolverErrorString(code), file, line);
        print_stacktrace();
        exit(code);
    }
}

#else
#define CUBLAS_CALL(ans) (ans)
#define CUDA_CALL(ans) (ans)
#define CUSOLVER_CALL(ans) (ans)
#define CUSPARSE_CALL(ans) (ans)
#endif

#define MAX_STREAMS 16



class GPU_Operations {
    cublasHandle_t handle;
    curandState* rng_state;
    cusolverDnHandle_t cudense_handle;
    cusparseHandle_t cusparse_handle;
    std::map<size_t, void*> buffer_map; // keeps track of buffers allocated for potrf
    int* devinfo; // cuSOLVER error reporting
    cudaStream_t streams[MAX_STREAMS];
    cusparseMatDescr_t descr;



public:
float* ones;

struct SparseMatrix {
	float *values;
	int *columns;
	int *rowPointers;
	int n; // number of rows
	int nnz; // number of nonzero elements
};

const SparseMatrix INVALID = {
	(float*)-1, (int*)-1, (int*)-1, 0, 0
};

static SparseMatrix create_sparse_matrix(const float* Xvals, const int* Xcols, const int *Xrowptr, int n, int m);
static void free_sparse_matrix(const SparseMatrix& x);

GPU_Operations(int n, int m, int k, unsigned long seed, int gpu_id);
~GPU_Operations();

float* to_device(const float* src, size_t size) const;
int* to_device(const int* src, size_t size) const;
SparseMatrix* to_device(const SparseMatrix* src, size_t size) const;

float* to_host(float* src, float* dst, size_t size) const {
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    free(src);
    return dst;
}

int* to_host(int* src, int* dst, size_t size) const {
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    free(src);
    return dst;
}

float* copy_to_host(const float* src, float* dst, size_t size) const {
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return dst;
}

int* copy_to_host(const int* src, int* dst, size_t size) const {
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return dst;
}

void set_stream(unsigned iterator) const {
    unsigned stream_id = iterator % MAX_STREAMS;
    CUBLAS_CALL(cublasSetStream_v2(handle, streams[stream_id]));
}

void synchronize_stream(unsigned iterator) const {
    unsigned stream_id = iterator % MAX_STREAMS;
    CUDA_CALL(cudaStreamSynchronize(streams[stream_id]));
}

void synchronize_all_streams() const {
    for (unsigned i = 0; i < MAX_STREAMS; ++i) {
        synchronize_stream(i);
    }
}

void default_stream() const {
    CUBLAS_CALL(cublasSetStream_v2(handle, NULL));
}

void gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
        const float *a, const int lda, const float *b, const int ldb, const float beta, float *c,
        const int ldc) const {
    cublasOperation_t ta = tolower(transa[0]) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t tb = tolower(transb[0]) == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CALL(cublasSgemm(handle, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

void gemm(const char *transa, const char *transb, const int m,
          const int n, const int k, const float alpha,
          const SparseMatrix* a, const int lda, const float *b,
          const int ldb, const float beta, float *c,
          const int ldc);

void gemm(const char *transa, const char *transb, const int m,
          const int n, const int k, const float alpha, const float *a,
          const int lda, const SparseMatrix* b, const int ldb,
          const float beta, float *c, const int ldc);

void dgmm(const char* mode, const int m, const int n, const float* A,
          int lda, const float* x, int incx, float* C,
          int ldc) const {
    cublasSideMode_t lr = mode[0] == 'l' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    CUBLAS_CALL(cublasSdgmm(handle, lr, m, n, A, lda, x, incx, C, ldc));
}

void symm(const char *side, const char *uplo, const int m, const int n,
          const float alpha, const float *a, const int lda, const float *b,
          const int ldb, const float beta, float *c, const int ldc) const {
    cublasSideMode_t s = tolower(side[0]) == 'l' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t ul = uplo_to_cublas(uplo);
    CUBLAS_CALL(cublasSsymm(handle, s, ul, m, n, &alpha,a, lda, b, ldb, &beta, c, ldc));
}

void axpy(const int n, const float alpha, const float* x, const int incx, float *y, const int incy) const {
    CUBLAS_CALL(cublasSaxpy(handle, n, &alpha, x, incx, y, incy));
}

int potrf(const char *uplo, int n, float* a, int lda) {
    cublasFillMode_t ul = uplo_to_cublas(uplo);
    int bufsize = 0;
    int info = 0;
    CUSOLVER_CALL(cusolverDnSpotrf_bufferSize(cudense_handle, ul, n, a, lda, &bufsize));

    float* buffer = (float*) get_buffer(bufsize * sizeof(float));

    CUSOLVER_CALL(cusolverDnSpotrf(cudense_handle, ul, n, a, lda, buffer, bufsize, devinfo));
    CUDA_CALL(cudaMemcpy(&info, devinfo, sizeof(int), cudaMemcpyDeviceToHost));
    return info;
}

void* get_buffer(size_t bufsize) {
    // See if we already have a buffer of correct size, otherwise allocate
    void* buffer = 0;
    auto it = buffer_map.find(bufsize);
    if (it != buffer_map.end()) {
        buffer = it->second;
    } else {
        buffer = malloc(bufsize);
        buffer_map[bufsize] = buffer;
    }
    return buffer;
}

int potrs(const char *uplo, int n, int nrhs, float * a, int lda, float *b, int ldb) const {
    int info;
    cublasFillMode_t ul = uplo_to_cublas(uplo);
    CUSOLVER_CALL(cusolverDnSpotrs(cudense_handle, ul, n, nrhs, a, lda, b, ldb, devinfo));
    CUDA_CALL(cudaMemcpy(&info, devinfo, sizeof(info), cudaMemcpyDeviceToHost));
    return info;
}

int posv(const char *uplo, int n, int nrhs, float * a, int lda, float *b, int ldb) {
    int info = potrf(uplo, n, a, lda);
    if (info == 0)
        info = potrs(uplo, n, nrhs, a, lda, b, ldb);
    return info;
}

void* memset(void* dest, int ch, size_t count) const {
    CUDA_CALL(cudaMemset(dest, ch, count));
    return dest;
}

float* memcpy(void* dest, const void *src, size_t count) const {
    CUDA_CALL(cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice));
    return 0;
}

void free(void* ptr) const {
    if (ptr != 0 && ptr != &INVALID) {
        CUDA_CALL(cudaFree(ptr));
    }
}

void free_devicememory(void* ptr) const {
    if (ptr != 0) {
        CUDA_CALL(cudaFree(ptr));
    }
}

void free_devicememory(SparseMatrix* matrix) {
    if (matrix != 0 && matrix != &INVALID) {
        free(matrix->columns);
        free(matrix->values);
        free(matrix->rowPointers);
        std::free(matrix);
    }
}

float* malloc(size_t size) const {
    float* retval = 0;
    cudaError_t err = cudaMalloc(&retval, size);
    CUDA_CALL(err);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        retval = 0;
    }
    return retval;
}

int* malloci(size_t size) const {
    int* retval = 0;
    cudaError_t err = cudaMalloc(&retval, size);
    CUDA_CALL(err);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        retval = 0;
    }
    return retval;
}

void fill_eye(float* X, unsigned n) const;
void fill(float* X, const unsigned size, const float value) const;
void maximum(float* x, const float value, const unsigned size) const;
void leaky_relu(float* x, const float value, const unsigned size) const;
void tanh(float* x, const unsigned size) const;
void sigmoid(float* x, const unsigned size) const;
void soft_threshold(float* x, const float alpha, const int size) const;
void invsqrt(float* s, const unsigned n) const;

void invert(float* X, const unsigned size) const;

void calculate_column_variance(const float* X, const unsigned nrows, const unsigned ncols, float* variances, float eps) const;
void scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
void scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const;
void dropout(float* X, const unsigned size, const float dropout_rate) const;
void add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const;
void add_gauss_noise(float* X, const unsigned size, const float noise_rate) const;

void calculate_column_variance(const SparseMatrix* X, const unsigned nrows, const unsigned ncols, float* variances, float eps);
void scale_columns(SparseMatrix* X, const unsigned nrows, const unsigned ncols, float* s) const;
void scale_rows(SparseMatrix* X, const unsigned nrows, const unsigned ncols, float* s) const;
void dropout(SparseMatrix* X, const unsigned size, const float dropout_rate) const;
void add_saltpepper_noise(SparseMatrix* X, const unsigned size, const float noise_rate) const;
void add_gauss_noise(SparseMatrix* X, const unsigned size, const float noise_rate) const;

template<typename T>
T init_invalid(void) {
    return (typeid(T) == typeid(SparseMatrix*) ? (T) &INVALID : (T) 0);
}

template<typename T>
T malloc_matrix(int rows, int cols) {
    return malloc_matrix(rows, cols, init_invalid<T>());
}

SparseMatrix* malloc_matrix(int rows, int cols, SparseMatrix* dummy) {
    SparseMatrix* matrix = (SparseMatrix*) std::malloc(sizeof(SparseMatrix));
    return matrix;
}

float* malloc_matrix(int rows, int cols, float *dummy) {
    return (float*) malloc(rows * cols * sizeof(float));
}

float *memcpy_matrix(float *dest, float *src, int nrows_to_copy, int src_ncol, int first_row = 0) const {
    return memcpy(dest, &src[first_row * src_ncol], nrows_to_copy * src_ncol * sizeof(float));
}

SparseMatrix* memcpy_matrix(SparseMatrix* dest, SparseMatrix* src, int nrows_to_copy, int src_ncol, int first_row = 0) const {
    int fromIndex = 0;
    int toIndex   = 0;
    CUDA_CALL(cudaMemcpy(&fromIndex, &src->rowPointers[first_row], sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&toIndex  , &src->rowPointers[first_row + nrows_to_copy], sizeof(int), cudaMemcpyDeviceToHost));

    dest->nnz = (toIndex - fromIndex);
    dest->n = nrows_to_copy;

    dest->values = malloc(dest->nnz * sizeof(float));
    dest->columns = malloci(dest->nnz * sizeof(int));
    dest->rowPointers = malloci((nrows_to_copy + 1) * sizeof(int));

    memcpy(dest->values, &src->values[fromIndex], dest->nnz * sizeof(float));
    memcpy(dest->columns, &src->columns[fromIndex], dest->nnz * sizeof(int));
    memcpy(dest->rowPointers, &src->rowPointers[first_row], (nrows_to_copy + 1) * sizeof(int));
    subtract_first_element(dest->rowPointers, nrows_to_copy + 1);

    return dest;
}

void subtract_first_element(int* a, unsigned len) const;

void free_batch(void *ptr) {
}

void free_batch(SparseMatrix* a) {
    // see get batch
    if (handle_valid(a)) {
        free(a->rowPointers);
        std::free(a);
    }
}

bool handle_valid(SparseMatrix* a) {
    return a != &INVALID;
}

float* get_batch(const float* X, int ncol, int batch_num, int batch_size) {
    /* return pointer */
    return (float*) &X[batch_num * batch_size * ncol];
}

SparseMatrix* get_batch(SparseMatrix* X, int ncol, int batch_num, int batch_size) {
    // ncol can be ignored
    // batch_size number of rows
    int from = batch_num * batch_size;
    int nrows = batch_size;

    SparseMatrix* dest = (SparseMatrix*) std::malloc(sizeof(SparseMatrix));
    int fromIndex = 0;
    int toIndex   = 0;
    CUDA_CALL(cudaMemcpy(&fromIndex, &X->rowPointers[from], sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&toIndex  , &X->rowPointers[from + nrows], sizeof(int), cudaMemcpyDeviceToHost));

    dest->nnz = (toIndex - fromIndex);
    dest->n = nrows;
    dest->values = &X->values[fromIndex];
    dest->columns = &X->columns[fromIndex];
    dest->rowPointers = malloci((nrows + 1) * sizeof(int));
    memcpy(dest->rowPointers, &X->rowPointers[from], (nrows + 1) * sizeof(int));
    subtract_first_element(dest->rowPointers, nrows + 1);
    return dest;
}

SparseMatrix* transpose(const SparseMatrix* x, int ncol) {
    SparseMatrix* t = (SparseMatrix*) std::malloc(sizeof(SparseMatrix));
    t->values = //(float*) get_buffer(x->nnz * sizeof(float));
            malloc(x->nnz * sizeof(float));
    t->columns = //(int*) get_buffer(x->nnz * sizeof(int));
            malloci(x->nnz * sizeof(int));
    t->rowPointers = //(int*) get_buffer((ncol + 1) * sizeof(int));
            malloci((ncol + 1) * sizeof(int));
    t->nnz = x->nnz;
    t->n = ncol;
    CUSPARSE_CALL(cusparseScsr2csc(cusparse_handle, x->n, ncol, x->nnz, x->values, x->rowPointers, x->columns, t->values,
            t->columns, t->rowPointers, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));

    return t;
}

// Useful for debugging
void printm(const char* name, const SparseMatrix *a, int n, int m) const {
    printf("%s\n", name);
    printMatrixSPM(a, n, m, 0);
}

void printm(const char* name, const float* a, int n, int m) const {
    printf("%s\n", name);
    printMatrixRM(a, n, m, 0);
}


void printMatrixCM(const float* a, int n, int m, const char* fmt) const;
void printMatrixRM(const float* a, int n, int m, const char* fmt) const;

void printMatrixSP(const SparseMatrix* a, const char* fmt) const;
void printMatrixRM(const SparseMatrix* a, int n, int m, const char* fmt) const {
    printMatrixSPM(a, n, m, fmt);
}

void printMatrixSPM(const SparseMatrix* a, int n, int m, const char* fmt) const;

void prints(const float* f, unsigned l) const {
    float* src = (float*) std::malloc(l * sizeof(float));
    copy_to_host(f, src, l * sizeof(float));
    for (unsigned i = 0; i < l; ++i) {
        printf("%f ", src[i]);
    }
    printf("\n");
    std::free(src);
}

void printsu(const int* f, unsigned l) const {
    int* src = (int*) std::malloc(l * sizeof(int));
    copy_to_host(f, src, l * sizeof(int));
    for (unsigned i = 0; i < l; ++i) {
        printf("%d ", src[i]);
    }
    printf("\n");
    std::free(src);
}
};
