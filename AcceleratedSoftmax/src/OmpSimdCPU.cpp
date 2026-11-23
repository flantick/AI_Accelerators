#include "OmpSimdCPU.h"

#include <algorithm>
#include <cmath>

void softmax_omp_simd(const float* x, float* y, std::size_t M, std::size_t N) {
    if (N == 0) return;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < M; ++i) {
        const float* row_in = x + i * N;
        float* row_out = y + i * N;

        const float max_row_value = *(std::max_element(row_in, row_in + N));

        #pragma omp simd
        for (std::size_t j = 0; j < N; ++j) {
            row_out[j] = std::exp(row_in[j] - max_row_value);
        }

        float sum_exp = 0.0f;
        for (std::size_t j = 0; j < N; ++j) {
            sum_exp += row_out[j];
        }

        const float inv_sum = 1.0f / sum_exp;
        #pragma omp simd
        for (std::size_t j = 0; j < N; ++j) {
            row_out[j] *= inv_sum;
        }
    }
}
