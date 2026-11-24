#ifndef SOFTMAXACCELERATED_BENCH_UTILS_H
#define SOFTMAXACCELERATED_BENCH_UTILS_H

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <string>
#include <vector>

namespace bench {

constexpr std::size_t kAlignment = 64;

class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t count) {
        std::size_t bytes = (count * sizeof(float) + kAlignment - 1) / kAlignment * kAlignment;
#ifdef _WIN32
        data_ = static_cast<float*>(_aligned_malloc(bytes, kAlignment));
#else
        data_ = static_cast<float*>(std::aligned_alloc(kAlignment, bytes));
#endif
        if (!data_) throw std::bad_alloc();
    }
    ~AlignedBuffer() {
#ifdef _WIN32
        _aligned_free(data_);
#else
        std::free(data_);
#endif
    }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    float* data() { return data_; }

private:
    float* data_ = nullptr;
};

inline void fill_uniform(float* p, std::size_t n, float lo, float hi, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (std::size_t i = 0; i < n; ++i) p[i] = dist(gen);
}

inline bool read_f32(const std::string& path, float* p, std::size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::size_t got = std::fread(p, sizeof(float), n, f);
    std::fclose(f);
    return got == n;
}

inline bool write_f32(const std::string& path, const float* p, std::size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::size_t put = std::fwrite(p, sizeof(float), n, f);
    std::fclose(f);
    return put == n;
}

struct TimingStats {
    int reps = 0;
    double median_ms = 0, min_ms = 0;
};

template <class F>
TimingStats measure_with(F&& timed_once, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) timed_once();

    std::vector<double> samples(reps);
    for (double& s : samples) s = timed_once();
    std::sort(samples.begin(), samples.end());

    const std::size_t n = samples.size();
    TimingStats st;
    st.reps = reps;
    st.median_ms = n % 2 ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    st.min_ms = samples.front();
    return st;
}

template <class F>
TimingStats measure(F&& fn, int warmup, int reps) {
    return measure_with([&] {
        const auto start = std::chrono::steady_clock::now();
        fn();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }, warmup, reps);
}

}

#endif
