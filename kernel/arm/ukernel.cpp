#include <arm_neon.h>

// Compute a dot-product between two int8 vectors using ARM NEON dot-product intrinsics.
int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    int32_t i = 0;

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