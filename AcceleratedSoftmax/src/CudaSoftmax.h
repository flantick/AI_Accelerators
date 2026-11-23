#ifndef SOFTMAXACCELERATED_CUDASOFTMAX_H
#define SOFTMAXACCELERATED_CUDASOFTMAX_H

#include <cstddef>
#include <string>

#include "bench_utils.h"

void softmax_cuda(const float* x, float* y, std::size_t M, std::size_t N);

bench::TimingStats bench_softmax_cuda(const float* x, float* y, std::size_t M, std::size_t N, int warmup, int reps);

std::string cuda_device_info_json();

#endif
