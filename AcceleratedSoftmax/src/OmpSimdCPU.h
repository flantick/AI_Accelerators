#ifndef SOFTMAXACCELERATED_OMPSIMDCPU_H
#define SOFTMAXACCELERATED_OMPSIMDCPU_H

#include <cstddef>

void softmax_omp_simd(const float* x, float* y, std::size_t M, std::size_t N);

#endif
