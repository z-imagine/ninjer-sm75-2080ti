#pragma once

#include "ops/common/memory.cuh"

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

#if defined(NINFER_SM75)
// ============================================================================
// sm75 (TU102) mma fallback.
//
// Ground truth for this build (CUDA 12.4 ptxas + runtime probes on the 2080Ti):
//   native mma.sync shapes: m16n8k8 f16 (f32 acc), m8n8k16 s8 (s32 acc) — both
//   runtime-verified correct.
//   ptxas rejects on sm_75 with "requires .target sm_80 or higher":
//   m16n8k16 f16, m16n8k16 bf16, tf32 m16n8k4/m16n8k8, s8 m16n8k16/m16n8k32.
//   fp8/nvfp4 block-scaled mma requires sm_89/sm_120a.
//
// Fragment layouts below are from the PTX ISA mma.sync "Matrix Fragments"
// sections. The m16n8k16 f16 split into two m16n8k8 calls was verified on this
// hardware with a random-input probe (max error 0 vs software reference) before
// this file was written.
// ============================================================================

// Gather 16 bf16 values (one A row or one B column) into f32. The 4-lane group
// starting at base_lane holds, for lane base+i: lo = elements 2i, 2i+1 and
// hi = elements 2i+8, 2i+9. This matches the PTX ISA m16n8k16 bf16 layout for
// both the A fragment (rows) and the B fragment (columns):
//   A: a0=(r,2t..2t+1) a1=(r+8,2t..2t+1) a2=(r,2t+8..2t+9) a3=(r+8,2t+8..2t+9)
//   B: b0=(k 2t..2t+1, n) b1=(k 2t+8..2t+9, n)
// bf16 -> f32 is exact (bf16 is f32 with the low 16 mantissa bits zeroed).
__device__ __forceinline__ void sm75_gather_bf16_16(float v[16], unsigned lo, unsigned hi,
                                                    int base_lane) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned l = __shfl_sync(0xffffffffu, lo, base_lane + i);
        const unsigned h = __shfl_sync(0xffffffffu, hi, base_lane + i);
        v[2 * i]     = __uint_as_float(l << 16);
        v[2 * i + 1] = __uint_as_float(l & 0xffff0000u);
        v[2 * i + 8]  = __uint_as_float(h << 16);
        v[2 * i + 9]  = __uint_as_float(h & 0xffff0000u);
    }
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    // No bf16 tensor cores on sm75: FMA emulation (design D2). bf16 -> f32 is
    // exact, products are exact in f32 (8+8 mantissa bits), accumulation is a
    // deterministic sequential k=0..15 dot product.
    const int lane = threadIdx.x & 31;
    const int r    = lane >> 2;
    const int t    = lane & 3;
    float ar[16], arb[16], bcol0[16], bcol1[16];
    sm75_gather_bf16_16(ar,    a0, a2, 4 * r);            // A[r][k],   k = 0..15
    sm75_gather_bf16_16(arb,   a1, a3, 4 * r);            // A[r+8][k], k = 0..15
    sm75_gather_bf16_16(bcol0, b0, b1, 4 * (2 * t));      // B[k][2t],  k = 0..15
    sm75_gather_bf16_16(bcol1, b0, b1, 4 * (2 * t + 1));  // B[k][2t+1], k = 0..15
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        c0 += ar[k] * bcol0[k];
        c1 += ar[k] * bcol1[k];
        c2 += arb[k] * bcol0[k];
        c3 += arb[k] * bcol1[k];
    }
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    // m16n8k16 f16 does not exist on sm75: two native m16n8k8 calls.
    // PTX ISA layouts map the registers directly:
    //   m16n8k16 A: a0=(r,2t..) a1=(r+8,2t..) a2=(r,2t+8..) a3=(r+8,2t+8..)
    //   m16n8k8  A: a0=(r,2t..) a1=(r+8,2t..) for k = 0..7
    //   B:        b0 = k 0..7, b1 = k 8..15 (both fragments, same layout)
    //   C/D:      identical 4-register layout in both shapes
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a2), "r"(a3), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    // m16n8k32 s8 does not exist on sm75: four native m8n8k16 calls.
    // PTX ISA layouts:
    //   m16n8k32 s8 A: a0=(r,4t..4t+3) a1=(r+8,4t..4t+3) a2=(r,16+4t..) a3=(r+8,16+4t..)
    //   m16n8k32 s8 B: b0=B[4t..4t+3][n] (k 0..15), b1=B[16+4t..][n] (k 16..31)
    //   m8n8k16  s8 A: (row, 4t..4t+3); B: B[4t..4t+3][n]; C/D: (row,2t),(row,2t+1)
    // so (a0,b0) -> rows 0-7 k 0-15 = c0,c1; (a1,b0) -> rows 8-15 k 0-15 = c2,c3;
    //    (a2,b1) -> rows 0-7 k 16-31 -> c0,c1; (a3,b1) -> rows 8-15 k 16-31 -> c2,c3.
    // Integer arithmetic: exact regardless of accumulation order.
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c2), "+r"(c3)
                 : "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a2), "r"(b1));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c2), "+r"(c3)
                 : "r"(a3), "r"(b1));
}

// sm75 has no FP8/NVFP4 tensor cores (sm_89/sm_120a). CMake excludes the
// consumers from the sm75 build (task 3.3); these trap loudly if one ever
// reaches here instead of silently producing garbage.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    (void)c0;
    (void)c1;
    (void)c2;
    (void)c3;
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    asm volatile("trap;");
}

// Round an f32 to TF32 precision (10-bit mantissa) with round-to-nearest-even,
// matching the hardware conversion applied to tf32 mma inputs on sm80+.
__device__ __forceinline__ float sm75_tf32_round(float f) {
    unsigned u = __float_as_uint(f);
    u          = (u + 0x0FFFu + ((u >> 13) & 1u)) & 0xFFFFE000u;
    return __uint_as_float(u);
}

// Gather 8 tf32 values (one A row or one B column) into f32. The 4-lane group
// starting at base_lane holds, for lane base+i: first = element i,
// second = element i+4. This matches the PTX ISA m16n8k8 tf32 layout for both
// fragments: A: a0=(r,t) a2=(r,t+4); B: b0=(t,n) b1=(t+4,n).
__device__ __forceinline__ void sm75_gather_tf32_8(float v[8], unsigned first, unsigned second,
                                                   int base_lane) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i]     = __uint_as_float(__shfl_sync(0xffffffffu, first, base_lane + i));
        v[i + 4] = __uint_as_float(__shfl_sync(0xffffffffu, second, base_lane + i));
    }
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    // No tf32 tensor cores on sm75 (ptxas: requires sm_80): FMA emulation
    // (design D2). Inputs are rounded to tf32 RNE first so the operand
    // precision matches the sm80+ hardware path; products are exact in f32
    // (11x11 significand bits); accumulation is a sequential k=0..7 dot.
    // PTX ISA m16n8k8 tf32 layout (lane: r = laneid>>2, t = laneid&3):
    //   A: a0=(r,t) a1=(r+8,t) a2=(r,t+4) a3=(r+8,t+4)
    //   B: b0=(t,r) b1=(t+4,r)
    //   D: c0=(r,2t) c1=(r,2t+1) c2=(r+8,2t) c3=(r+8,2t+1)
    const int lane = threadIdx.x & 31;
    const int r    = lane >> 2;
    const int t    = lane & 3;
    float ar[8], arb[8], bcol0[8], bcol1[8];
    sm75_gather_tf32_8(ar,    a0, a2, 4 * r);            // A[r][k],   k = 0..7
    sm75_gather_tf32_8(arb,   a1, a3, 4 * r);            // A[r+8][k], k = 0..7
    sm75_gather_tf32_8(bcol0, b0, b1, 4 * (2 * t));      // B[k][2t],  k = 0..7
    sm75_gather_tf32_8(bcol1, b0, b1, 4 * (2 * t + 1));  // B[k][2t+1], k = 0..7
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float ar0  = sm75_tf32_round(ar[k]);
        const float arb0 = sm75_tf32_round(arb[k]);
        const float b00  = sm75_tf32_round(bcol0[k]);
        const float b01  = sm75_tf32_round(bcol1[k]);
        c0 += ar0 * b00;
        c1 += ar0 * b01;
        c2 += arb0 * b00;
        c3 += arb0 * b01;
    }
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    (void)c0;
    (void)c1;
    (void)c2;
    (void)c3;
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    (void)sfa;
    (void)sfb;
    asm volatile("trap;");
}

#else  // !NINFER_SM75

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// On RTX 5090, block-scaled FP8 avoids the plain form's half-rate FP32 accumulation.
// UE8M0 0x7f is exactly 1: preserve the raw E4M3 dot product and keep the callers'
// existing row scales in FP32 after the MMA. No operand repacking is needed.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

#endif  // NINFER_SM75

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

}  // namespace ninfer::ops
