#ifndef SOFTMAXACCELERATED_OMPCPU_H
#define SOFTMAXACCELERATED_OMPCPU_H

#include <cstddef>

void softmax_omp(const float* x, float* y, std::size_t M, std::size_t N);

#endif
