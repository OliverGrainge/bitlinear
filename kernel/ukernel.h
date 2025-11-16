// Common interface for architecture-specific int8 dot-product micro-kernels.
// This header declares the micro-kernel interface that is implemented
// separately for different architectures (ARM NEON, x86 AVX2, and scalar fallback).

#pragma once

#include <cstdint>

// Single dot product between two int8 vectors.
// Returns the sum of element-wise products: sum(a[i] * b[i]) for i in [0, length)
int32_t i8dot(const int8_t* a, const int8_t* b, int32_t length);

// 1x4 micro-kernel: compute 4 dot-products between `a` and {b0,b1,b2,b3}.
// Accumulates results into c0, c1, c2, c3 (caller should initialize to 0).
// This is useful for GEMM-style operations where one vector is reused.
void i8dot_1x4(
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
);