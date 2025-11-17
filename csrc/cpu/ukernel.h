// Implementation of int8 dot-product micro-kernels for different architectures.
// The appropriate backend is selected at compile time via architecture macros.

#pragma once

#include <cstdint>
#include <iostream>

// ---------------------------------------------------------------------------
// ARM NEON IMPLEMENTATION (macOS arm64, AArch64)
// ---------------------------------------------------------------------------
#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>

inline int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    int32_t i = 0;
    
    // Process 64 bytes per iteration (4x16 bytes)
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

    // Combine accumulators
    acc0 = vaddq_s32(acc0, acc1);
    acc2 = vaddq_s32(acc2, acc3);
    int32x4_t acc = vaddq_s32(acc0, acc2);

    int32_t sum = vaddvq_s32(acc);

    // Handle remaining elements
    for (; i < length; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }

    return sum;
}

inline void i8dot_1x4(
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
    
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    int32_t i = 0;

    // Process 64 bytes per iteration
    for (; i + 63 < length; i += 64) {
        // Load A vectors (reused for all 4 dot products)
        int8x16_t va0 = vld1q_s8(a + i);
        int8x16_t va1 = vld1q_s8(a + i + 16);
        int8x16_t va2 = vld1q_s8(a + i + 32);
        int8x16_t va3 = vld1q_s8(a + i + 48);

        // Load B0 vectors
        int8x16_t vb0_0 = vld1q_s8(b0 + i);
        int8x16_t vb0_1 = vld1q_s8(b0 + i + 16);
        int8x16_t vb0_2 = vld1q_s8(b0 + i + 32);
        int8x16_t vb0_3 = vld1q_s8(b0 + i + 48);

        // Load B1 vectors
        int8x16_t vb1_0 = vld1q_s8(b1 + i);
        int8x16_t vb1_1 = vld1q_s8(b1 + i + 16);
        int8x16_t vb1_2 = vld1q_s8(b1 + i + 32);
        int8x16_t vb1_3 = vld1q_s8(b1 + i + 48);

        // Load B2 vectors
        int8x16_t vb2_0 = vld1q_s8(b2 + i);
        int8x16_t vb2_1 = vld1q_s8(b2 + i + 16);
        int8x16_t vb2_2 = vld1q_s8(b2 + i + 32);
        int8x16_t vb2_3 = vld1q_s8(b2 + i + 48);

        // Load B3 vectors
        int8x16_t vb3_0 = vld1q_s8(b3 + i);
        int8x16_t vb3_1 = vld1q_s8(b3 + i + 16);
        int8x16_t vb3_2 = vld1q_s8(b3 + i + 32);
        int8x16_t vb3_3 = vld1q_s8(b3 + i + 48);

        // Compute dot products for all 4 chunks
        acc0 = vdotq_s32(acc0, va0, vb0_0);
        acc1 = vdotq_s32(acc1, va0, vb1_0);
        acc2 = vdotq_s32(acc2, va0, vb2_0);
        acc3 = vdotq_s32(acc3, va0, vb3_0);

        acc0 = vdotq_s32(acc0, va1, vb0_1);
        acc1 = vdotq_s32(acc1, va1, vb1_1);
        acc2 = vdotq_s32(acc2, va1, vb2_1);
        acc3 = vdotq_s32(acc3, va1, vb3_1);

        acc0 = vdotq_s32(acc0, va2, vb0_2);
        acc1 = vdotq_s32(acc1, va2, vb1_2);
        acc2 = vdotq_s32(acc2, va2, vb2_2);
        acc3 = vdotq_s32(acc3, va2, vb3_2);

        acc0 = vdotq_s32(acc0, va3, vb0_3);
        acc1 = vdotq_s32(acc1, va3, vb1_3);
        acc2 = vdotq_s32(acc2, va3, vb2_3);
        acc3 = vdotq_s32(acc3, va3, vb3_3);
    }

    // Accumulate results
    c0 += vaddvq_s32(acc0);
    c1 += vaddvq_s32(acc1);
    c2 += vaddvq_s32(acc2);
    c3 += vaddvq_s32(acc3);

    // Handle remaining elements
    for (; i < length; ++i) {
        int32_t aa = static_cast<int32_t>(a[i]);
        c0 += aa * static_cast<int32_t>(b0[i]);
        c1 += aa * static_cast<int32_t>(b1[i]);
        c2 += aa * static_cast<int32_t>(b2[i]);
        c3 += aa * static_cast<int32_t>(b3[i]);
    }
}

// ---------------------------------------------------------------------------
// x86 AVX2 IMPLEMENTATION (no VNNI required)
// ---------------------------------------------------------------------------
#elif defined(__AVX2__)

#include <immintrin.h>

inline int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length) {
    
    __m256i vacc = _mm256_setzero_si256();
    int32_t i = 0;

    // Process 32 bytes per iteration (2x16 bytes widened to 16-bit)
    for (; i + 31 < length; i += 32) {
        // First 16 bytes
        __m128i va0_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb0_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m256i va0_16 = _mm256_cvtepi8_epi16(va0_8);
        __m256i vb0_16 = _mm256_cvtepi8_epi16(vb0_8);
        __m256i prod0 = _mm256_madd_epi16(va0_16, vb0_16); // 16 int8 → 8 int32
        vacc = _mm256_add_epi32(vacc, prod0);

        // Next 16 bytes
        __m128i va1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16));
        __m128i vb1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16));
        __m256i va1_16 = _mm256_cvtepi8_epi16(va1_8);
        __m256i vb1_16 = _mm256_cvtepi8_epi16(vb1_8);
        __m256i prod1 = _mm256_madd_epi16(va1_16, vb1_16);
        vacc = _mm256_add_epi32(vacc, prod1);
    }

    // Horizontal sum of vacc (8 x int32)
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(vacc),
                                   _mm256_extracti128_si256(vacc, 1));
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    int32_t sum = _mm_cvtsi128_si32(sum128);

    // Handle remaining tail elements
    for (; i < length; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }

    return sum;
}

inline void i8dot_1x4(
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
    
    __m256i vacc0 = _mm256_setzero_si256();
    __m256i vacc1 = _mm256_setzero_si256();
    __m256i vacc2 = _mm256_setzero_si256();
    __m256i vacc3 = _mm256_setzero_si256();

    int32_t i = 0;
    
    // Process 32 bytes per iteration
    for (; i + 31 < length; i += 32) {
        // First 16 bytes of A
        __m128i va0_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m256i va0_16 = _mm256_cvtepi8_epi16(va0_8);

        // First 16 bytes of each B
        __m128i vb0_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b0 + i));
        __m128i vb1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b1 + i));
        __m128i vb2_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b2 + i));
        __m128i vb3_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b3 + i));

        __m256i vb0_16 = _mm256_cvtepi8_epi16(vb0_8);
        __m256i vb1_16 = _mm256_cvtepi8_epi16(vb1_8);
        __m256i vb2_16 = _mm256_cvtepi8_epi16(vb2_8);
        __m256i vb3_16 = _mm256_cvtepi8_epi16(vb3_8);

        vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(va0_16, vb0_16));
        vacc1 = _mm256_add_epi32(vacc1, _mm256_madd_epi16(va0_16, vb1_16));
        vacc2 = _mm256_add_epi32(vacc2, _mm256_madd_epi16(va0_16, vb2_16));
        vacc3 = _mm256_add_epi32(vacc3, _mm256_madd_epi16(va0_16, vb3_16));

        // Next 16 bytes of A
        __m128i va1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16));
        __m256i va1_16 = _mm256_cvtepi8_epi16(va1_8);

        // Next 16 bytes of each B
        __m128i vc0_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b0 + i + 16));
        __m128i vc1_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b1 + i + 16));
        __m128i vc2_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b2 + i + 16));
        __m128i vc3_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b3 + i + 16));

        __m256i vc0_16 = _mm256_cvtepi8_epi16(vc0_8);
        __m256i vc1_16 = _mm256_cvtepi8_epi16(vc1_8);
        __m256i vc2_16 = _mm256_cvtepi8_epi16(vc2_8);
        __m256i vc3_16 = _mm256_cvtepi8_epi16(vc3_8);

        vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(va1_16, vc0_16));
        vacc1 = _mm256_add_epi32(vacc1, _mm256_madd_epi16(va1_16, vc1_16));
        vacc2 = _mm256_add_epi32(vacc2, _mm256_madd_epi16(va1_16, vc2_16));
        vacc3 = _mm256_add_epi32(vacc3, _mm256_madd_epi16(va1_16, vc3_16));
    }

    // Horizontal sum helper
    auto hsum = [](__m256i v) -> int32_t {
        __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v),
                                       _mm256_extracti128_si256(v, 1));
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        return _mm_cvtsi128_si32(sum128);
    };

    c0 += hsum(vacc0);
    c1 += hsum(vacc1);
    c2 += hsum(vacc2);
    c3 += hsum(vacc3);

    // Handle remaining tail elements
    for (; i < length; ++i) {
        int32_t aa = static_cast<int32_t>(a[i]);
        c0 += aa * static_cast<int32_t>(b0[i]);
        c1 += aa * static_cast<int32_t>(b1[i]);
        c2 += aa * static_cast<int32_t>(b2[i]);
        c3 += aa * static_cast<int32_t>(b3[i]);
    }
}

// ---------------------------------------------------------------------------
// PORTABLE SCALAR FALLBACK with OpenMP SIMD
// ---------------------------------------------------------------------------
#else

#include <omp.h>

inline int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length) {
    
    int32_t sum = 0;
    
    #pragma omp simd reduction(+:sum)
    for (int32_t i = 0; i < length; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    
    return sum;
}

inline void i8dot_1x4(
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
    
    // Note: We can't use reduction on multiple variables easily with OpenMP SIMD,
    // so we compute each dot product separately with SIMD pragmas
    
    #pragma omp simd reduction(+:c0)
    for (int32_t i = 0; i < length; ++i) {
        c0 += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b0[i]);
    }
    
    #pragma omp simd reduction(+:c1)
    for (int32_t i = 0; i < length; ++i) {
        c1 += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b1[i]);
    }
    
    #pragma omp simd reduction(+:c2)
    for (int32_t i = 0; i < length; ++i) {
        c2 += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b2[i]);
    }
    
    #pragma omp simd reduction(+:c3)
    for (int32_t i = 0; i < length; ++i) {
        c3 += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b3[i]);
    }
}

#endif  // architecture selection