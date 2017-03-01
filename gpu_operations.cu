/*
Copyright © 2015-2017 Thomas Unterthiner
Additional Contributions by Thomas Adler, Balázs Bencze
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <stdexcept>

#include "gpu_operations.h"

static const int RNG_THREADS = 128;
static const int RNG_BLOCKS = 128;

// taken from PyCUDA
void get_grid_sizes(int problemsize, int* blocks, int* threads) {
    int min_threads = 32;
    int max_threads = 256;
    int max_blocks = 384;

    if (problemsize < min_threads) {
        *blocks = 1;
        *threads = min_threads;
    } else if (problemsize < max_blocks * min_threads) {
        *blocks = (problemsize + min_threads - 1) / min_threads;
        *threads = min_threads;
    } else if (problemsize < max_blocks * max_threads) {
        *blocks = max_blocks;
        int grp = (problemsize + min_threads - 1) / min_threads;
        *threads = ((grp + max_blocks - 1) / max_blocks) * min_threads;
    } else {
        *blocks = max_blocks;
        *threads = max_threads;
    }
}

__global__ void setup_rng(curandState* rng_state, unsigned long seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &rng_state[tid]);
}

__global__ void dropout_eltw(float* x, const unsigned size, const float dropout_rate, curandState* rng_state) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    curandState localState = rng_state[tid];
    for (unsigned i = tid; i < size; i += num_threads)
        x[i] = (curand_uniform(&localState) < dropout_rate) ? 0.0 : x[i];
    rng_state[tid] = localState;
}

__global__ void saltpepper_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    curandState localState = rng_state[tid];
    for (unsigned i = tid; i < size; i += num_threads)
        if (curand_uniform(&localState) < noise_rate) {
            x[i] = (curand_uniform(&localState) < 0.5f) ? 0.0f : 1.0f;
        }
    rng_state[tid] = localState;

}

__global__ void gauss_noise_eltw(float* x, const unsigned size, const float noise_rate, curandState* rng_state) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    curandState localState = rng_state[tid];
    for (unsigned i = tid; i < size; i += num_threads)
        x[i] += curand_normal(&localState) * noise_rate;
    rng_state[tid] = localState;

}

__global__ void leaky_relu_eltw(float* x, const float value, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = (x[i] < 0.0f) ? x[i] * value : x[i];
    }
}

__global__ void maximum_eltw(float* x, const float value, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = fmaxf(x[i], value);
    }
}

__global__ void sigmoid_eltw(float* x, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = 1 / (1 + __expf(-x[i]));
    }
}

__global__ void tanh_eltw(float* x, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = tanhf(x[i]);
    }
}

__global__ void softthreshold_eltw(float* x, float alpha, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        const float f = x[i];
        x[i] = f > 0 ? fmaxf(0., f - alpha) : fminf(0., f + alpha);
    }
}

__global__ void fill_eltw(float* x, const unsigned size, const float value) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = value;
    }
}

__global__ void invert_eltw(float* x, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x * blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = 1.0f / x[i];
    }
}

__global__ void col_variance_kernel(const float* X, float* var, const unsigned nrows, const unsigned ncols, float eps) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < ncols; i += num_threads) {
        var[i] = 0.0;
        for (unsigned j = 0; j < nrows; ++j) {
            var[i] += X[j * ncols + i];
        }
        float m = var[i] / nrows;
        var[i] = 0.0;
        for (unsigned j = 0; j < nrows; ++j) {
            float tmp = X[j * ncols + i] - m;
            var[i] += tmp * tmp;
        }
        var[i] /= nrows;
        var[i] += eps; // for numerical stability
    }
}

__global__ void invsqrt_eltw(float* x, const unsigned k) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < k; i += num_threads) {
        x[i] = rsqrtf(x[i] + 1e-8);
    }
}

__global__ void scale_columns_kernel(float* X, float* a, const unsigned nrows, const unsigned ncols) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < ncols * nrows; i += num_threads) {
        X[i] *= a[i % ncols];
    }
}

__global__ void scale_rows_kernel(float* X, float* a, const unsigned nrows, const unsigned ncols) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < ncols * nrows; i += num_threads) {
        X[i] *= a[i / ncols];
    }
}

__global__ void subtract_first_kernel(int* x, const unsigned len) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    const unsigned elem = x[0];
    for (unsigned i = tid; i < len; i += num_threads) {
        x[i] -= elem;
    }
}

__global__ void sparse_col_variance_kernel(const GPU_Operations::SparseMatrix X, float* var, const unsigned nrows,
        const unsigned ncols, float eps) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < ncols; i += num_threads) {
        var[i] = 0.0;
        for (unsigned j = 0; j < X.nnz; ++j) {
            if (X.columns[j] == i) {
                var[i] += X.values[j];
            }
        }
        float m = var[i] / nrows;
        var[i] = 0.0;
        unsigned nonzero_per_column = 0;
        for (unsigned j = 0; j < X.nnz; ++j) {
            if (X.columns[j] == i) {
                float tmp = X.values[j] - m;
                var[i] += tmp * tmp;
                nonzero_per_column++;
            }
        }
        var[i] += (nrows - nonzero_per_column) * (m * m);
        var[i] /= nrows;
        var[i] += eps; // for numerical stability
    }
}

__global__ void sparse_row_variance_kernel(const GPU_Operations::SparseMatrix X, float* var, const unsigned nrows,
        const unsigned ncols, float eps) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < nrows; i += num_threads) {
        var[i] = 0.0;
        int from = X.rowPointers[i];
        int to = X.rowPointers[i + 1];
        for (int j = from; j < to; ++j) {
            var[i] += X.values[j];
        }
        float m = var[i] / ncols;
        var[i] = 0.0;
        for (int j = from; j < to; ++j) {
            float tmp = X.values[j] - m;
            var[i] += tmp * tmp;
        }
        var[i] += (ncols - to + from) * (m * m);
        var[i] /= ncols;
        var[i] += eps; // for numerical stability
    }
}

__global__ void sparse_scale_columns_kernel(GPU_Operations::SparseMatrix X, float* a, const unsigned nrows, const unsigned ncols) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < X.nnz; i += num_threads) {
        X.values[i] *= a[X.columns[i]];
    }
}

__global__ void sparse_scale_rows_kernel(GPU_Operations::SparseMatrix X, float* a) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < X.m; i += num_threads) {
        for (unsigned j = X.rowPointers[i]; j < X.rowPointers[i + 1]; ++j) {
            X.values[j] *= a[i];
        }
    }
}

GPU_Operations::GPU_Operations(const int n, const int m, const int k, unsigned long seed, int gpu_id) {
    // if no GPU was specified, try to pick the best one automatically
    if (gpu_id < 0) {
        gpu_id = 0;
        int num_devices, device;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 1) {
            size_t max_freememory = 0;
            for (device = 0; device < num_devices; device++) {
                size_t free, total;
                cudaSetDevice(device);
                cudaMemGetInfo(&free, &total);
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, device);
                //printf("Found device %d (%s) with %d MiB of free memory\n",
                //    device, prop.name, free / (1024l*1024l));
                if (free > max_freememory) {
                    max_freememory = free;
                    gpu_id = device;
                }
                cudaDeviceReset();
            }
        }
    }
    assert(gpu_id >= 0);
    cudaSetDevice(gpu_id);

    // the following call does not work if the current process has already
    // called into librfn previously. Then, this call will return
    // cudaErrorSetOnActiveProcess. Resetting the device won't work either,
    // because then the subsequent cublasCreate call will just fail with
    // CUBLAS_STATUS_NOT_INITIALIZED. I don't know why any of this is happening
    //CUDA_CALL(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char* errmsg = cublasErrorString(status);
        fprintf(stderr, "CUBLAS initialization error: %s\n", errmsg);
        cudaDeviceReset();
        throw std::runtime_error(errmsg);
    }
    CUSOLVER_CALL(cusolverDnCreate(&cudense_handle));
    CUDA_CALL(cudaMalloc(&rng_state, RNG_BLOCKS * RNG_THREADS * sizeof(curandState)));
    setup_rng<<<RNG_BLOCKS, RNG_THREADS>>>(rng_state, seed);
    int ones_size = n > k ? n : k;
    ones = (float*) malloc(ones_size * sizeof(float));
    fill(ones, ones_size, 1.0f);
    CUDA_CALL(cudaMalloc(&devinfo, sizeof(int)));

    cusparseStatus_t sp_status = cusparseCreate(&cusparse_handle);
    if (sp_status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "cuSparse: %d\n", sp_status);
        cudaDeviceReset();
        throw std::runtime_error("cuSparse error");
    }

    for (int i = 0; i < MAX_STREAMS; i++) {
        CUDA_CALL(cudaStreamCreate(&streams[i]));
    }

    CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
    CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
}

GPU_Operations::~GPU_Operations() {
    free(devinfo);
    free(ones);
    for (auto i : buffer_map) {
        free(i.second);
    }
    CUSOLVER_CALL(cusolverDnDestroy(cudense_handle));
    CUBLAS_CALL(cublasDestroy(handle));
    for (int i = 0; i < MAX_STREAMS; i++) {
        CUDA_CALL(cudaStreamSynchronize(streams[i]));
        CUDA_CALL(cudaStreamDestroy(streams[i]));
    }
    CUSPARSE_CALL(cusparseDestroyMatDescr(descr));
}

GPU_Operations::SparseMatrix GPU_Operations::create_sparse_matrix(const float* Xvals, const int* Xcols, const int *Xrowptr, int n, int m){
    SparseMatrix X = {(float*) Xvals, (int*) Xcols, (int*) Xrowptr, n, Xrowptr[n]};
    return X;
}


float* GPU_Operations::to_device(const float* src, size_t size) const {
    float* dst = 0;
    CUDA_CALL(cudaMalloc(&dst, size));
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return dst;
}

int* GPU_Operations::to_device(const int* src, size_t size) const {
    int* dst = 0;
    CUDA_CALL(cudaMalloc(&dst, size));
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return dst;
}

GPU_Operations::SparseMatrix* GPU_Operations::to_device(const SparseMatrix* src, size_t size) const {
    SparseMatrix* dst = (SparseMatrix*) std::malloc(sizeof(SparseMatrix));

    dst->values = to_device(src->values, src->nnz * sizeof(float));
    dst->columns = to_device(src->columns, src-> nnz * sizeof(int));
    dst->rowPointers = to_device(src->rowPointers, (src->m + 1) * sizeof(int));
    dst->m = src->m;
    dst->nnz = src->nnz;

    return dst;
}

void GPU_Operations::fill(float* X, const unsigned size, const float value) const {
    int threads, blocks;
    get_grid_sizes(size, &threads, &blocks);
    fill_eltw<<<blocks, threads>>>(X, size, value);
    assert(!cudaGetLastError());
}

void GPU_Operations::dropout(float* X, const unsigned size, const float dropout_rate) const {
    dropout_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, dropout_rate, rng_state);
    assert(!cudaGetLastError());
}

void GPU_Operations::add_gauss_noise(float* X, const unsigned size, const float noise_rate) const {
    gauss_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, noise_rate, rng_state);
    assert(!cudaGetLastError());
}

void GPU_Operations::add_saltpepper_noise(float* X, const unsigned size, const float noise_rate) const {
    saltpepper_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, noise_rate, rng_state);
    assert(!cudaGetLastError());
}

void GPU_Operations::invert(float* X, const unsigned size) const {
    int threads, blocks;
    get_grid_sizes(size, &threads, &blocks);
    invert_eltw<<<blocks, threads>>>(X, size);
    assert(!cudaGetLastError());
}

void GPU_Operations::maximum(float* x, const float value, const unsigned size) const {
    int threads, blocks;
    get_grid_sizes(size, &threads, &blocks);
    maximum_eltw<<<blocks, threads>>>(x, value, size);
    assert(!cudaGetLastError());
}

void GPU_Operations::leaky_relu(float* x, const float value, const unsigned size) const {
    int threads, blocks;
    get_grid_sizes(size, &threads, &blocks);
    leaky_relu_eltw<<<blocks, threads>>>(x, value, size);
    assert(!cudaGetLastError());
}

void GPU_Operations::sigmoid(float* x, const unsigned size) const {
    int threads, blocks;
    get_grid_sizes(size, &threads, &blocks);
    sigmoid_eltw<<<blocks, threads>>>(x, size);
    assert(!cudaGetLastError());
}

void GPU_Operations::tanh(float* x, const unsigned size) const {
    int threads, blocks;
    get_grid_sizes(size, &threads, &blocks);
    tanh_eltw<<<blocks, threads>>>(x, size);
    assert(!cudaGetLastError());
}

void GPU_Operations::soft_threshold(float* x, const float alpha, const int size) const {
    int threads, blocks;
    get_grid_sizes(size, &threads, &blocks);
    softthreshold_eltw<<<blocks, threads>>>(x, alpha, size);
    assert(!cudaGetLastError());
}

void GPU_Operations::fill_eye(float* X, unsigned n) const {
    memset(X, 0, n * n * sizeof(float));
    axpy(n, 1.0f, ones, 0, X, n + 1);
}

void GPU_Operations::calculate_column_variance(const float* X, const unsigned nrows, const unsigned ncols,
        float* variance, float eps) const {
    int threads, blocks;
    get_grid_sizes(ncols, &threads, &blocks);
    col_variance_kernel<<<threads, blocks>>>(X, variance, nrows, ncols, eps);
}

void GPU_Operations::invsqrt(float* s, const unsigned n) const {
    int t, b;
    get_grid_sizes(n, &t, &b);
    invsqrt_eltw<<<t, b>>>(s, n);
}

void GPU_Operations::scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const {

    int threads, blocks;
    get_grid_sizes(ncols * nrows, &threads, &blocks);
    scale_columns_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}

void GPU_Operations::scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const {
    int threads, blocks;
    get_grid_sizes(ncols * nrows, &threads, &blocks);
    scale_rows_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}

void GPU_Operations::subtract_first_element(int* a, unsigned len) const {
    int threads, blocks;
    get_grid_sizes(len, &threads, &blocks);
    subtract_first_kernel<<<threads, blocks>>>(a, len);
}

void GPU_Operations::calculate_column_variance(const SparseMatrix* X, const unsigned nrows, const unsigned ncols,
        float* variance, float eps) {
    int threads, blocks;
    SparseMatrix* x_transpose = transpose(X, ncols);
    get_grid_sizes(nrows, &threads, &blocks);
    sparse_row_variance_kernel<<<threads, blocks>>>(*x_transpose, variance, ncols, nrows, eps);
    free(x_transpose->columns);
    free(x_transpose->values);
    free(x_transpose->rowPointers);
    std::free(x_transpose);

}

void GPU_Operations::scale_columns(SparseMatrix* X, const unsigned nrows, const unsigned ncols, float* s) const {

    int threads, blocks;
    get_grid_sizes(X->nnz, &threads, &blocks);
    sparse_scale_columns_kernel<<<threads, blocks>>>(*X, s, nrows, ncols);
}

void GPU_Operations::scale_rows(SparseMatrix* X, const unsigned nrows, const unsigned ncols, float* s) const {
    int threads, blocks;
    get_grid_sizes(X->m, &threads, &blocks);
    sparse_scale_rows_kernel<<<threads, blocks>>>(*X, s);
}

void GPU_Operations::dropout(SparseMatrix* X, const unsigned size, const float dropout_rate) const {
    dropout_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X->values, size, dropout_rate, rng_state);
    assert(!cudaGetLastError());
}

void GPU_Operations::add_gauss_noise(SparseMatrix* X, const unsigned size, const float noise_rate) const {
    gauss_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X->values, size, noise_rate, rng_state);
    assert(!cudaGetLastError());
}

void GPU_Operations::add_saltpepper_noise(SparseMatrix* X, const unsigned size, const float noise_rate) const {
    saltpepper_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X->values, size, noise_rate, rng_state);
    assert(!cudaGetLastError());
}

void GPU_Operations::gemm(const char *transa, const char *transb, const int m, const int n, const int k, const float alpha,
        const SparseMatrix* a, const int lda, const float *b, const int ldb, const float beta, float *c,
        const int ldc) {
    cusparseOperation_t opA = op_to_cusparse(transa);
    cusparseOperation_t opB = op_to_cusparse(transb);

    SparseMatrix* row_major_a = transpose(a, opA != CUSPARSE_OPERATION_NON_TRANSPOSE ? k : m);

    int ncol_a = k;
    if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE) {
        ncol_a = a->m;
    }

    CUSPARSE_CALL(cusparseScsrmm2(cusparse_handle, opA, opB, row_major_a->m, n, ncol_a,
            row_major_a->nnz, &alpha, descr, row_major_a->values, row_major_a->rowPointers, row_major_a->columns, b, ldb, &beta, c, ldc));


    free(row_major_a->columns);
    free(row_major_a->values);
    free(row_major_a->rowPointers);
    std::free(row_major_a);
}

/*void GPU_Operations::gemm(const char *transa, const char *transb, const int m, const int n, const int k,
            const float alpha, const float *a, const int lda, const SparseMatrix* b, const int ldb,
            const float beta, float *c, const int ldc) {
    cusparseOperation_t opA = op_to_cusparse(transa);
    cusparseOperation_t opB = op_to_cusparse(transb);
    SparseMatrix* b_trans;

    if (opB != CUSPARSE_OPERATION_NON_TRANSPOSE) {
        b_trans = transpose(b, n);
    } else {
        b_trans = (SparseMatrix*) std::malloc(sizeof(SparseMatrix));
        b_trans->values = b->values;
        b_trans->columns = b->columns;
        b_trans->rowPointers = b->rowPointers;
        b_trans->m = b->m;
        b_trans->nnz = b->nnz;
    }

    int m_a = m; // number of rows of A
    int n_a = k; // number of columns of A
    if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE) {
        m_a = k;
        n_a = m;
    }

    int bufsize;
    CUSPARSE_CALL(cusparseSgemvi_bufferSize(cusparse_handle, opA, m_a, n_a, b_trans->nnz, &bufsize));
    void* buffer = get_buffer(bufsize);

    int* row_pointers = (int*) std::malloc((b_trans->m + 1) * sizeof(int));
    copy_to_host(b_trans->rowPointers, row_pointers, (b_trans->m + 1) * sizeof(int));

    for(unsigned r = 0; r < b_trans->m; ++r) {
        int row_pointer = row_pointers[r];
        int nnz = row_pointers[r + 1] - row_pointer;

        set_stream(r);

        if (nnz == 0) {
            CUBLAS_CALL(cublasSscal_v2(handle, n, &beta, &c[r * ldc], 1));
        } else if (nnz > 0) {
            CUSPARSE_CALL(cusparseSgemvi(cusparse_handle, opA, m_a, n_a, &alpha, a, lda, nnz,
                    &b_trans->values[row_pointer], &b_trans->columns[row_pointer], &beta, &c[r * ldc], CUSPARSE_INDEX_BASE_ZERO, buffer));
        } else {
            printf("Internal error");
            exit(1);
        }
    }

    synchronize_all_streams();
    default_stream();

    if (opB != CUSPARSE_OPERATION_NON_TRANSPOSE) {
        free(b_trans->values);
        free(b_trans->columns);
        free(b_trans->rowPointers);
    }
    std::free(b_trans);
    std::free(row_pointers);
}*/

void GPU_Operations::gemm(const char *transa, const char *transb, const int m, const int n, const int k,
            const float alpha, const float *a, const int lda, const SparseMatrix* b, const int ldb,
            const float beta, float *c, const int ldc) {
    cusparseOperation_t opA = op_to_cusparse(transa);
    cusparseOperation_t opB = op_to_cusparse(transb);
    SparseMatrix* b2;
    float alpha_t = 1.0f;
    float beta_t = 0.0f;

    //3)
    int b2_ncol = 0;
    if (opB != CUSPARSE_OPERATION_NON_TRANSPOSE) {
        b2 = transpose(b, n);
        b2_ncol = b->m;
    } else {
        b2 = (SparseMatrix*) std::malloc(sizeof(SparseMatrix));
        b2->values = b->values;
        b2->columns = b->columns;
        b2->rowPointers = b->rowPointers;
        b2->m = b->m;
        b2->nnz = b->nnz;
        b2_ncol = k;
    }
    //4)
    float* c2 = (float*) get_buffer(m*n * sizeof(float));
    memcpy(c2, c, m*n * sizeof(float));
    if (beta != 0.0f) {
        CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, &alpha_t, c, ldc, &beta_t, NULL, 0, c2, ldc));
    }

    // 4.5
    cusparseOperation_t opA2;
    if (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) {
        opA2 = CUSPARSE_OPERATION_TRANSPOSE;
    } else {
        opA2 = CUSPARSE_OPERATION_NON_TRANSPOSE;
    }

    //5)
    CUSPARSE_CALL(cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, opA2, b2->m, m, b2_ncol, b2->nnz, &alpha, descr,
            b2->values, b2->rowPointers, b2->columns, a, lda, &beta, c2, b2->m));

    //6
    CUBLAS_CALL(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha_t, c2, b2->m, &beta_t, (float*)0, b2->m, c, ldc));
    if (opB != CUSPARSE_OPERATION_NON_TRANSPOSE) {
        free(b2->columns);
        free(b2->values);
        free(b2->rowPointers);
    }
    std::free(b2);

}

// Debugging
void GPU_Operations::printMatrixRM(const float* a, int n, int m, const char* fmt) const {
    const char* format = fmt == 0 ? "%1.3f " : fmt;
    size_t size = n * m * sizeof(float);
    float* tmp = (float*) std::malloc(size);
    CUDA_CALL(cudaMemcpy(tmp, a, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j)
            printf(format, tmp[i * m + j]);
        printf("\n");
    }
    printf("\n");
    std::free(tmp);
}

void GPU_Operations::printMatrixCM(const float* a, int n, int m, const char* fmt) const {
    const char* format = fmt == 0 ? "%1.3f " : fmt;
    size_t size = n * m * sizeof(float);
    float* tmp = (float*) std::malloc(size);
    CUDA_CALL(cudaMemcpy(tmp, a, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j)
            printf(format, tmp[i + j * n]);
        printf("\n");
    }
    printf("\n");
    std::free(tmp);
}

void GPU_Operations::printMatrixSP(const SparseMatrix *a, const char* fmt) const {
    const char* format = fmt == 0 ? "%1.3f " : fmt;
    size_t size_values = a->nnz * sizeof(float);
    size_t size_columns = a->nnz * sizeof(int);
    size_t size_pointers = (a->m + 1)* sizeof(int);

    float* tmp_vals = (float*) std::malloc(size_values);
    int* tmp_cols = (int*) std::malloc(size_columns);
    int* tmp_pointers = (int*) std::malloc(size_pointers);

    CUDA_CALL(cudaMemcpy(tmp_vals, a->values, size_values, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tmp_cols, a->columns, size_columns, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tmp_pointers, a->rowPointers, size_pointers, cudaMemcpyDeviceToHost));

    printf("values: ");
    for (int i = 0; i < a->nnz; i++) {
        printf(format, tmp_vals[i]);
    }
    printf("\npointers: ");
    for (int i = 0; i <  a->m + 1; i++) {
        printf("%d ", tmp_pointers[i]);
    }
    printf("\ncolumns: ");
    for (int i = 0; i < a->nnz; i++) {
        printf("%d ", tmp_cols[i]);
    }
    printf("\n");
    std::free(tmp_vals);
    std::free(tmp_cols);
    std::free(tmp_pointers);
}

void GPU_Operations::printMatrixSPM(const SparseMatrix *a, int n, int m, const char* fmt) const {
    const char* format = fmt == 0 ? "%1.3f " : fmt;
    size_t size_values = a->nnz * sizeof(float);
    size_t size_columns = a->nnz * sizeof(int);
    size_t size_pointers = (a->m + 1)* sizeof(int);

    float* tmp_vals = (float*) std::malloc(size_values);
    int* tmp_cols = (int*) std::malloc(size_columns);
    int* tmp_pointers = (int*) std::malloc(size_pointers);

    CUDA_CALL(cudaMemcpy(tmp_vals, a->values, size_values, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tmp_cols, a->columns, size_columns, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tmp_pointers, a->rowPointers, size_pointers, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        int rowPointer = tmp_pointers[i];
        int nnz = tmp_pointers[i + 1] - rowPointer;
        int found = 0;
        for (int j = 0; j < m; j++) {
            if (found < nnz) {
                if (j == tmp_cols[rowPointer + found]) {
                    printf(format, tmp_vals[rowPointer + found]);
                    found++;
                } else {
                    printf(format, 0.0f);
                }
            } else {
                printf(format, 0.0f);
            }
        }
        printf("\n");
    }
    printf("\n");
    std::free(tmp_vals);
    std::free(tmp_cols);
    std::free(tmp_pointers);
}
