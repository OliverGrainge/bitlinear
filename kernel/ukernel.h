// Common interface for architecture-specific int8 dot-product micro-kernels.
// NOTE:
//   To avoid linker / dlopen issues across platforms (macOS arm64 & x86),
//   we provide header-only, static inline implementations here, with the
//   appropriate backend selected at compile time via architecture macros.
//   This ensures:
//     - No undefined symbols at load time
//     - No duplicate global symbols when building on different architectures
//     - Single source of truth for the micro-kernel interface

#pragma once

#include <cstdint>
#include <iostream>

// ---------------------------------------------------------------------------
// ARM NEON IMPLEMENTATION (macOS arm64, AArch64)
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>

// Single dot product between two int8 vectors using ARM NEON dot-product.
static inline int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    int32_t i = 0;
    std::cout << "i8dot_neon" << std::endl;
    for (; i + 63 < length; i += 64) {
        int8x16_t va0 = vld1q_s8(a + i);
        int8x16_t vb0 = vld1q_s8(b + i);
        int8x16_t va1 = vld1q_s8(a + i + 16);
        int8x16_t vb1 = vld1q_s8(b + i + 16);
        int8x16_t va2 = vld1q_s8(a + i + 32);
        int8x16_t vb2 = vld1q_s8(b + i + 32);
        int8x16_t va3 = vld1q_s8(a + i + 48);
        int8x16_t vb3 = vld1q_s8(b + i + 48);

        acc0 = vdotq_s32(acc0, va0, vb0);
        acc1 = vdotq_s32(acc1, va1, vb1);
        acc2 = vdotq_s32(acc2, va2, vb2);
        acc3 = vdotq_s32(acc3, va3, vb3);
    }

    acc0 = vaddq_s32(acc0, acc1);
    acc2 = vaddq_s32(acc2, acc3);
    int32x4_t acc = vaddq_s32(acc0, acc2);

    int32_t sum = vaddvq_s32(acc);

    for (; i < length; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }

    return sum;
}

// 1x4 micro-kernel: compute 4 dot-products between `a` and {b0,b1,b2,b3}.
static inline void i8dot_1x4(
    const int8_t* __restrict a,
    const int8_t* __restrict b0,
    const int8_t* __restrict b1,
    const int8_t* __restrict b2,
    const int8_t* __restrict b3,
    int32_t& c0,
    int32_t& c1,
    int32_t& c2,
    int32_t& c3,
    int32_t length
) {
    std::cout << "i8dot_1x4_neon" << std::endl;
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    int32_t i = 0;

    for (; i + 63 < length; i += 64) {
        int8x16_t va0 = vld1q_s8(a  + i);
        int8x16_t vb0 = vld1q_s8(b0 + i);
        int8x16_t vb1 = vld1q_s8(b1 + i);
        int8x16_t vb2 = vld1q_s8(b2 + i);
        int8x16_t vb3 = vld1q_s8(b3 + i);

        int8x16_t va1 = vld1q_s8(a  + i + 16);
        int8x16_t vc0 = vld1q_s8(b0 + i + 16);
        int8x16_t vc1 = vld1q_s8(b1 + i + 16);
        int8x16_t vc2 = vld1q_s8(b2 + i + 16);
        int8x16_t vc3 = vld1q_s8(b3 + i + 16);

        int8x16_t va2 = vld1q_s8(a  + i + 32);
        int8x16_t vd0 = vld1q_s8(b0 + i + 32);
        int8x16_t vd1 = vld1q_s8(b1 + i + 32);
        int8x16_t vd2 = vld1q_s8(b2 + i + 32);
        int8x16_t vd3 = vld1q_s8(b3 + i + 32);

        int8x16_t va3 = vld1q_s8(a  + i + 48);
        int8x16_t ve0 = vld1q_s8(b0 + i + 48);
        int8x16_t ve1 = vld1q_s8(b1 + i + 48);
        int8x16_t ve2 = vld1q_s8(b2 + i + 48);
        int8x16_t ve3 = vld1q_s8(b3 + i + 48);

        acc0 = vdotq_s32(acc0, va0, vb0);
        acc1 = vdotq_s32(acc1, va0, vb1);
        acc2 = vdotq_s32(acc2, va0, vb2);
        acc3 = vdotq_s32(acc3, va0, vb3);

        acc0 = vdotq_s32(acc0, va1, vc0);
        acc1 = vdotq_s32(acc1, va1, vc1);
        acc2 = vdotq_s32(acc2, va1, vc2);
        acc3 = vdotq_s32(acc3, va1, vc3);

        acc0 = vdotq_s32(acc0, va2, vd0);
        acc1 = vdotq_s32(acc1, va2, vd1);
        acc2 = vdotq_s32(acc2, va2, vd2);
        acc3 = vdotq_s32(acc3, va2, vd3);

        acc0 = vdotq_s32(acc0, va3, ve0);
        acc1 = vdotq_s32(acc1, va3, ve1);
        acc2 = vdotq_s32(acc2, va3, ve2);
        acc3 = vdotq_s32(acc3, va3, ve3);
    }

    c0 += vaddvq_s32(acc0);
    c1 += vaddvq_s32(acc1);
    c2 += vaddvq_s32(acc2);
    c3 += vaddvq_s32(acc3);

    for (; i < length; ++i) {
        int32_t aa = static_cast<int32_t>(a[i]);
        c0 += aa * static_cast<int32_t>(b0[i]);
        c1 += aa * static_cast<int32_t>(b1[i]);
        c2 += aa * static_cast<int32_t>(b2[i]);
        c3 += aa * static_cast<int32_t>(b3[i]);
    }
}

// ---------------------------------------------------------------------------
// x86 AVX-VNNI IMPLEMENTATION
// ---------------------------------------------------------------------------
#elif defined(__AVX2__) && defined(__AVX512VNNI__) || defined(__AVXVNNI__) || defined(__AVX512VNNI)

#include <immintrin.h>

// Scalar dot product helper.
static inline int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length) {
    std::cout << "i8dot_avx2" << std::endl;
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    int32_t i = 0;

    // Process 128 bytes at a time (4x32 bytes)
    for (; i + 127 < length; i += 128) {
        __m256i va0 = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb0 = _mm256_loadu_si256((__m256i*)(b + i));
        __m256i va1 = _mm256_loadu_si256((__m256i*)(a + i + 32));
        __m256i vb1 = _mm256_loadu_si256((__m256i*)(b + i + 32));
        __m256i va2 = _mm256_loadu_si256((__m256i*)(a + i + 64));
        __m256i vb2 = _mm256_loadu_si256((__m256i*)(b + i + 64));
        __m256i va3 = _mm256_loadu_si256((__m256i*)(a + i + 96));
        __m256i vb3 = _mm256_loadu_si256((__m256i*)(b + i + 96));

        acc0 = _mm256_dpbssd_epi32(acc0, va0, vb0);
        acc1 = _mm256_dpbssd_epi32(acc1, va1, vb1);
        acc2 = _mm256_dpbssd_epi32(acc2, va2, vb2);
        acc3 = _mm256_dpbssd_epi32(acc3, va3, vb3);
    }

    // Combine accumulators
    acc0 = _mm256_add_epi32(acc0, acc1);
    acc2 = _mm256_add_epi32(acc2, acc3);
    __m256i acc = _mm256_add_epi32(acc0, acc2);

    // Horizontal sum of 8 int32 values
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    int32_t sum = _mm_cvtsi128_si32(sum128);

    // Handle remainder
    for (; i < length; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }

    return sum;
}

// 1x4 micro-kernel for computing 4 dot-products in parallel
static inline void i8dot_1x4(
    const int8_t* __restrict a,
    const int8_t* __restrict b0,
    const int8_t* __restrict b1,
    const int8_t* __restrict b2,
    const int8_t* __restrict b3,
    int32_t& c0,
    int32_t& c1,
    int32_t& c2,
    int32_t& c3,
    int32_t length
) {
    std::cout << "i8dot_1x4_avx2" << std::endl;
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    int32_t i = 0;

    // Process 128 bytes at a time (4x32 bytes)
    for (; i + 127 < length; i += 128) {
        __m256i va0 = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb0 = _mm256_loadu_si256((__m256i*)(b0 + i));
        __m256i vb1 = _mm256_loadu_si256((__m256i*)(b1 + i));
        __m256i vb2 = _mm256_loadu_si256((__m256i*)(b2 + i));
        __m256i vb3 = _mm256_loadu_si256((__m256i*)(b3 + i));

        __m256i va1 = _mm256_loadu_si256((__m256i*)(a + i + 32));
        __m256i vc0 = _mm256_loadu_si256((__m256i*)(b0 + i + 32));
        __m256i vc1 = _mm256_loadu_si256((__m256i*)(b1 + i + 32));
        __m256i vc2 = _mm256_loadu_si256((__m256i*)(b2 + i + 32));
        __m256i vc3 = _mm256_loadu_si256((__m256i*)(b3 + i + 32));

        __m256i va2 = _mm256_loadu_si256((__m256i*)(a + i + 64));
        __m256i vd0 = _mm256_loadu_si256((__m256i*)(b0 + i + 64));
        __m256i vd1 = _mm256_loadu_si256((__m256i*)(b1 + i + 64));
        __m256i vd2 = _mm256_loadu_si256((__m256i*)(b2 + i + 64));
        __m256i vd3 = _mm256_loadu_si256((__m256i*)(b3 + i + 64));

        __m256i va3 = _mm256_loadu_si256((__m256i*)(a + i + 96));
        __m256i ve0 = _mm256_loadu_si256((__m256i*)(b0 + i + 96));
        __m256i ve1 = _mm256_loadu_si256((__m256i*)(b1 + i + 96));
        __m256i ve2 = _mm256_loadu_si256((__m256i*)(b2 + i + 96));
        __m256i ve3 = _mm256_loadu_si256((__m256i*)(b3 + i + 96));

        acc0 = _mm256_dpbssd_epi32(acc0, va0, vb0);
        acc1 = _mm256_dpbssd_epi32(acc1, va0, vb1);
        acc2 = _mm256_dpbssd_epi32(acc2, va0, vb2);
        acc3 = _mm256_dpbssd_epi32(acc3, va0, vb3);

        acc0 = _mm256_dpbssd_epi32(acc0, va1, vc0);
        acc1 = _mm256_dpbssd_epi32(acc1, va1, vc1);
        acc2 = _mm256_dpbssd_epi32(acc2, va1, vc2);
        acc3 = _mm256_dpbssd_epi32(acc3, va1, vc3);

        acc0 = _mm256_dpbssd_epi32(acc0, va2, vd0);
        acc1 = _mm256_dpbssd_epi32(acc1, va2, vd1);
        acc2 = _mm256_dpbssd_epi32(acc2, va2, vd2);
        acc3 = _mm256_dpbssd_epi32(acc3, va2, vd3);

        acc0 = _mm256_dpbssd_epi32(acc0, va3, ve0);
        acc1 = _mm256_dpbssd_epi32(acc1, va3, ve1);
        acc2 = _mm256_dpbssd_epi32(acc2, va3, ve2);
        acc3 = _mm256_dpbssd_epi32(acc3, va3, ve3);
    }

    // Horizontal sum for each accumulator
    auto hsum = [](__m256i v) -> int32_t {
        __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        return _mm_cvtsi128_si32(sum128);
    };

    c0 += hsum(acc0);
    c1 += hsum(acc1);
    c2 += hsum(acc2);
    c3 += hsum(acc3);

    // Handle remainder
    for (; i < length; ++i) {
        int32_t aa = static_cast<int32_t>(a[i]);
        c0 += aa * static_cast<int32_t>(b0[i]);
        c1 += aa * static_cast<int32_t>(b1[i]);
        c2 += aa * static_cast<int32_t>(b2[i]);
        c3 += aa * static_cast<int32_t>(b3[i]);
    }
}

// ---------------------------------------------------------------------------
// PORTABLE SCALAR FALLBACK (no NEON / no AVX-VNNI)
// ---------------------------------------------------------------------------
#else

// Simple scalar implementations that work everywhere.
static inline int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length) {
    int32_t sum = 0;
    for (int32_t i = 0; i < length; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return sum;
}

static inline void i8dot_1x4(
    const int8_t* __restrict a,
    const int8_t* __restrict b0,
    const int8_t* __restrict b1,
    const int8_t* __restrict b2,
    const int8_t* __restrict b3,
    int32_t& c0,
    int32_t& c1,
    int32_t& c2,
    int32_t& c3,
    int32_t length
) {
    for (int32_t i = 0; i < length; ++i) {
        int32_t aa = static_cast<int32_t>(a[i]);
        c0 += aa * static_cast<int32_t>(b0[i]);
        c1 += aa * static_cast<int32_t>(b1[i]);
        c2 += aa * static_cast<int32_t>(b2[i]);
        c3 += aa * static_cast<int32_t>(b3[i]);
    }
}

#endif  // architecture selection
