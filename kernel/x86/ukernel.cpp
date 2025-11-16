#include <immintrin.h>

// Compute a dot-product between two int8 vectors using x86 AVX-VNNI intrinsics.
// Requires AVX-VNNI support (Intel Cascade Lake or newer, AMD Zen 4 or newer)
int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length) {
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

        // _mm256_dpbssd_epi32: dot product of 4 signed bytes, accumulate to 32-bit
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