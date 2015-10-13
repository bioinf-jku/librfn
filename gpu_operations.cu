/*
Copyright © 2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <stdexcept>

#include "gpu_operations.h"

static const int RNG_THREADS = 128;
static const int RNG_BLOCKS = 128;
/*
cublasHandle_t GPU_Operations::handle;
float* GPU_Operations::ones = 0;
curandState* GPU_Operations::rng_state = 0;
cudaStream_t* GPU_Operations::streams = 0;
*/

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


__global__ void setup_rng(curandState* rng_state, unsigned long seed)
{
    const int tid = blockIdx.x*blockDim.x+threadIdx.x;
    curand_init(seed, tid, 0, &rng_state[tid]);
}


__global__ void dropout_eltw(float* x, const unsigned size,
                             const float dropout_rate,
                             curandState* rng_state) {
    const unsigned tid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    curandState localState = rng_state[tid];
    for (unsigned i = tid; i < size; i += num_threads)
        x[i] = (curand_uniform(&localState) < dropout_rate) ? 0.0 : x[i];
    rng_state[tid] = localState;
}


__global__ void saltpepper_noise_eltw(float* x, const unsigned size,
                             const float noise_rate,
                             curandState* rng_state) {
    const unsigned tid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    curandState localState = rng_state[tid];
    for (unsigned i = tid; i < size; i += num_threads)
        if (curand_uniform(&localState) < noise_rate) {
            x[i] = (curand_uniform(&localState) < 0.5f) ? 0.0f : 1.0f;
        }
    rng_state[tid] = localState;

}


__global__ void gauss_noise_eltw(float* x, const unsigned size,
                             const float noise_rate,
                             curandState* rng_state) {
    const unsigned tid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    curandState localState = rng_state[tid];
    for (unsigned i = tid; i < size; i += num_threads)
            x[i] += curand_normal(&localState) * noise_rate ;
    rng_state[tid] = localState;

}


__global__ void leaky_relu_eltw(float* x, const float value, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
         x[i] = (x[i] < 0.0f) ? x[i] * value :  x[i];
    }
}


__global__ void maximum_eltw(float* x, const float value, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = fmaxf(x[i], value);
    }
}


__global__ void sigmoid_eltw(float* x, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
         x[i] = 1 / (1 + __expf(-x[i]));
    }
}


__global__ void tanh_eltw(float* x, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = tanhf(x[i]);
    }
}


__global__ void softthreshold_eltw(float* x, float alpha, const unsigned size) {
   const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
   const unsigned num_threads = gridDim.x*blockDim.x;
   for (unsigned i = tid; i < size; i += num_threads) {
       const float f = x[i];
       x[i] = f > 0 ? fmaxf(0., f - alpha) : fminf(0., f + alpha);
   }
}


__global__ void fill_eltw(float* x, const unsigned size, const float value) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = value;
    }
}


__global__ void invert_eltw(float* x, const unsigned size) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = gridDim.x*blockDim.x;
    for (unsigned i = tid; i < size; i += num_threads) {
        x[i] = 1.0f / x[i];
    }
}


__global__ void col_variance_kernel(const float* X, float* var,
                                    const unsigned nrows,
                                    const unsigned ncols) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < ncols; i += num_threads) {
            var[i] = 0.0;
        for (unsigned j = 0; j < nrows; ++j) {
            var[i] += X[j*ncols + i];
        }
        float m = var[i] / nrows;
        var[i] = 0.0;
        for (unsigned j = 0; j < nrows; ++j) {
            float tmp = X[j*ncols + i] - m;
            var[i] += tmp*tmp;
        }
        var[i] /= nrows;
    }
}


__global__ void invsqrt_eltw(float* x, const unsigned k) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned num_threads = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < k; i += num_threads) {
        x[i] = (x[i] > 1e-7) ? rsqrtf(x[i]) : 1.0;
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


GPU_Operations::GPU_Operations(const int n, const int m, const int k,
                                 unsigned long seed, int gpu_id) {

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
    CUDA_CALL(cudaMalloc(&rng_state, RNG_BLOCKS*RNG_THREADS*sizeof(curandState)));
    setup_rng<<<RNG_BLOCKS, RNG_THREADS>>>(rng_state, seed);
    int ones_size = n > k ? n : k;
    ones = malloc(ones_size*sizeof(float));
    fill(ones, ones_size, 1.0f);
    CUDA_CALL(cudaMalloc(&devinfo, sizeof(int)));
}


GPU_Operations::~GPU_Operations() {
    free(devinfo);
    free(ones);
    for (auto i : buffer_map) {
        free(i.second);
    }
    CUSOLVER_CALL(cusolverDnDestroy(cudense_handle));
    CUBLAS_CALL(cublasDestroy(handle));
}


float* GPU_Operations::to_device(const float* src, size_t size) const {
    float* dst = 0;
    CUDA_CALL(cudaMalloc(&dst, size));
    CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return dst;
}


void GPU_Operations::fill(float* X, const unsigned size, const float value) const {
    int threads, blocks;
    get_grid_sizes(size, &threads, &blocks);
    fill_eltw<<<blocks, threads>>>(X, size, value);
    assert(!cudaGetLastError());
}


void GPU_Operations::dropout(float* X, const unsigned size,
                                    const float dropout_rate) const {
    dropout_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, dropout_rate, rng_state);
    assert(!cudaGetLastError());
}


void GPU_Operations::add_gauss_noise(float* X, const unsigned size,
                                    const float noise_rate) const {
    gauss_noise_eltw<<<RNG_BLOCKS, RNG_THREADS>>>(X, size, noise_rate, rng_state);
    assert(!cudaGetLastError());
}


void GPU_Operations::add_saltpepper_noise(float* X, const unsigned size,
                                    const float noise_rate) const {
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


void GPU_Operations::soft_threshold(float* x, const float alpha, const unsigned size) const {
   int threads, blocks;
   get_grid_sizes(size, &threads, &blocks);
   softthreshold_eltw<<<blocks, threads>>>(x, alpha, size);
   assert(!cudaGetLastError());
}

void GPU_Operations::fill_eye(float* X, unsigned n) const {
    memset(X, 0, n*n*sizeof(float));
    axpy(n, 1.0f, ones, 0, X, n+1);
}


void GPU_Operations::calculate_column_variance(const float* X, const unsigned nrows,
                                               const unsigned ncols, float* variance) {
    int threads, blocks;
    get_grid_sizes(ncols, &threads, &blocks);
    col_variance_kernel<<<threads, blocks>>>(X, variance, nrows, ncols);
}


void GPU_Operations::invsqrt(float* s, const unsigned n) const {
    int t, b;
    get_grid_sizes(n, &t, &b);
    invsqrt_eltw<<<t, b>>>(s, n);
}

void GPU_Operations::scale_columns(float* X, const unsigned nrows, const unsigned ncols, float* s) const {

    int threads, blocks;
    get_grid_sizes(ncols*nrows, &threads, &blocks);
    scale_columns_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}


void GPU_Operations::scale_rows(float* X, const unsigned nrows, const unsigned ncols, float* s) const {
    int threads, blocks;
    get_grid_sizes(ncols*nrows, &threads, &blocks);
    scale_rows_kernel<<<threads, blocks>>>(X, s, nrows, ncols);
}


void GPU_Operations::printMatrixRM(const float* a, int n, int m, const char* fmt) {
    const char* format = fmt == 0 ? "%1.3f " : fmt;
    size_t size = n*m*sizeof(float);
    float* tmp = (float*) std::malloc(size);
    CUDA_CALL(cudaMemcpy(tmp, a, size, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n; ++i) {
		for (int j =0 ; j < m; ++j)
			printf(format, tmp[i*m + j]);
		printf("\n");
	}
    printf("\n");
    std::free(tmp);
}


void GPU_Operations::printMatrixCM(const float* a, int n, int m, const char* fmt) {
    const char* format = fmt == 0 ? "%1.3f " : fmt;
    size_t size = n*m*sizeof(float);
    float* tmp = (float*) std::malloc(size);
    CUDA_CALL(cudaMemcpy(tmp, a, size, cudaMemcpyDeviceToHost));
	for (int i = 0; i < n; ++i) {
		for (int j =0 ; j < m; ++j)
			printf(format, tmp[i + j*n]);
		printf("\n");
	}
    printf("\n");
    std::free(tmp);
}
