#ifndef SOFTMAXACCELERATED_NAIVECPU_H
#define SOFTMAXACCELERATED_NAIVECPU_H

#include <cstddef>

void softmax_naive(const float* x, float* y, std::size_t M, std::size_t N);

#endif
