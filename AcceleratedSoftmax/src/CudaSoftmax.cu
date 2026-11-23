#include "CudaSoftmax.h"

#include <cfloat>
#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

namespace {

constexpr int block_size = 256;
constexpr int warp_size = 32;
constexpr unsigned all_lanes = 0xffffffff;

__device__ float block_max(float value) {
    __shared__ float warp_values[warp_size];
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int warps = blockDim.x / warp_size;

    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(all_lanes, value, offset));
    }
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    value = lane < warps ? warp_values[lane] : -FLT_MAX;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(all_lanes, value, offset));
    }
    __syncthreads();
    return value;
}

__device__ float block_sum(float value) {
    __shared__ float warp_values[warp_size];
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int warps = blockDim.x / warp_size;

    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(all_lanes, value, offset);
    }
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();

    value = lane < warps ? warp_values[lane] : 0.0f;
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(all_lanes, value, offset);
    }
    __syncthreads();
    return value;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int N) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const std::size_t row_start = static_cast<std::size_t>(row) * N;
    const float* row_in = x + row_start;
    float* row_out = y + row_start;

    float max_row_value = -FLT_MAX;
    for (int col = thread; col < N; col += blockDim.x) {
        max_row_value = fmaxf(max_row_value, row_in[col]);
    }
    max_row_value = block_max(max_row_value);

    float sum_exp = 0.0f;
    for (int col = thread; col < N; col += blockDim.x) {
        sum_exp += expf(row_in[col] - max_row_value);
    }
    sum_exp = block_sum(sum_exp);

    const float inv_sum = 1.0f / sum_exp;
    for (int col = thread; col < N; col += blockDim.x) {
        row_out[col] = expf(row_in[col] - max_row_value) * inv_sum;
    }
}

void check_cuda(cudaError_t error, const char* call) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(error));
    }
}

#define CUDA_CHECK(call) check_cuda((call), #call)

struct GpuData {
    float* x = nullptr;
    float* y = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~GpuData() {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (y) cudaFree(y);
        if (x) cudaFree(x);
    }
};

bench::TimingStats run_on_gpu(const float* x, float* y, std::size_t M, std::size_t N, int warmup, int reps) {
    const std::size_t bytes = M * N * sizeof(float);
    GpuData gpu;
    CUDA_CHECK(cudaMalloc(&gpu.x, bytes));
    CUDA_CHECK(cudaMalloc(&gpu.y, bytes));
    CUDA_CHECK(cudaMemcpy(gpu.x, x, bytes, cudaMemcpyHostToDevice));

    const auto launch = [&] {
        softmax_kernel<<<static_cast<unsigned>(M), block_size>>>(gpu.x, gpu.y, static_cast<int>(N));
        CUDA_CHECK(cudaGetLastError());
    };

    bench::TimingStats stats;
    if (reps > 0) {
        CUDA_CHECK(cudaEventCreate(&gpu.start));
        CUDA_CHECK(cudaEventCreate(&gpu.stop));
        stats = bench::measure_with([&] {
            CUDA_CHECK(cudaEventRecord(gpu.start));
            launch();
            CUDA_CHECK(cudaEventRecord(gpu.stop));
            CUDA_CHECK(cudaEventSynchronize(gpu.stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, gpu.start, gpu.stop));
            return static_cast<double>(ms);
        }, warmup, reps);
    } else {
        launch();
    }

    CUDA_CHECK(cudaMemcpy(y, gpu.y, bytes, cudaMemcpyDeviceToHost));
    return stats;
}

#undef CUDA_CHECK

}

void softmax_cuda(const float* x, float* y, std::size_t M, std::size_t N) {
    run_on_gpu(x, y, M, N, 0, 0);
}

bench::TimingStats bench_softmax_cuda(const float* x, float* y, std::size_t M, std::size_t N, int warmup, int reps) {
    return run_on_gpu(x, y, M, N, warmup, reps);
}

std::string cuda_device_info_json() {
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess) return "null";
    int runtime = 0;
    int driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    char json[512];
    std::snprintf(json, sizeof json, "{\"name\":\"%s\",\"cc\":\"%d.%d\",\"cuda_runtime\":%d,\"cuda_driver\":%d}",
                  device.name, device.major, device.minor, runtime, driver);
    return json;
}
