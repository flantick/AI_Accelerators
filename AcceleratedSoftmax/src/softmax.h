#ifndef SOFTMAXACCELERATED_SOFTMAX_H
#define SOFTMAXACCELERATED_SOFTMAX_H

#include <cstddef>

using SoftmaxFn = void (*)(const float* x, float* y, std::size_t M, std::size_t N);

struct SoftmaxImpl {
    const char* name;
    SoftmaxFn fn;
};

#endif
