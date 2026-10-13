// Fragment-layout self-test for the mma wrappers in src/ops/common/mma.cuh
// (task 2.2). For each mma shape the fragment registers are built on the host
// from a random integer matrix exactly as the PTX ISA "Matrix Fragments"
// sections prescribe, the wrapper is invoked on a single warp, and each lane's
// c0..c3 is compared bit-exactly against a CPU dot-product reference.
//
// Integer-valued inputs keep every product and the full sum exactly
// representable in the accumulator type (f32 for f16/bf16/tf32, s32 for s8),
// so the verdict is independent of hardware accumulation order:
//   - on sm75 this validates the fallback (native m16n8k8 / m8n8k16 splits +
//     FMA emulation, design D2/D8);
//   - on sm86/sm120a the same test validates the native wrappers.

#include "ops/common/mma.cuh"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Frag4 {
    std::uint32_t r[4];
};
struct Frag2 {
    std::uint32_t r[2];
};

inline std::uint32_t pack2(std::uint16_t e0, std::uint16_t e1) {
    return std::uint32_t(e0) | (std::uint32_t(e1) << 16);
}

inline std::uint32_t pack4_i8(std::int8_t e0, std::int8_t e1, std::int8_t e2, std::int8_t e3) {
    return std::uint32_t(std::uint8_t(e0)) | (std::uint32_t(std::uint8_t(e1)) << 8) |
           (std::uint32_t(std::uint8_t(e2)) << 16) | (std::uint32_t(std::uint8_t(e3)) << 24);
}

inline std::uint16_t enc_f16(long long x) {
    return __half_as_ushort(__float2half_rn(static_cast<float>(x)));
}

inline std::uint32_t enc_tf32_bits(long long x) {
    float f = static_cast<float>(x);
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

inline int frag_r(int lane) { return lane >> 2; }
inline int frag_t(int lane) { return lane & 3; }

// ---- single-warp kernels ------------------------------------------------------

__global__ void kernel_bf16(const Frag4* a, const Frag2* b, float* out) {
    const int lane = threadIdx.x;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    ops::mma_bf16(c[0], c[1], c[2], c[3], a[lane].r[0], a[lane].r[1], a[lane].r[2], a[lane].r[3],
                  b[lane].r[0], b[lane].r[1]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[4 * lane + i] = c[i];
}

__global__ void kernel_f16(const Frag4* a, const Frag2* b, float* out) {
    const int lane = threadIdx.x;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    ops::mma_f16(c[0], c[1], c[2], c[3], a[lane].r[0], a[lane].r[1], a[lane].r[2], a[lane].r[3],
                 b[lane].r[0], b[lane].r[1]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[4 * lane + i] = c[i];
}

__global__ void kernel_s8(const Frag4* a, const Frag2* b, int* out) {
    const int lane = threadIdx.x;
    int c[4] = {0, 0, 0, 0};
    ops::mma_s8(c[0], c[1], c[2], c[3], a[lane].r[0], a[lane].r[1], a[lane].r[2], a[lane].r[3],
                b[lane].r[0], b[lane].r[1]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[4 * lane + i] = c[i];
}

__global__ void kernel_tf32(const Frag4* a, const Frag2* b, float* out) {
    const int lane = threadIdx.x;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    ops::mma_tf32_bits(c[0], c[1], c[2], c[3], a[lane].r[0], a[lane].r[1], a[lane].r[2], a[lane].r[3],
                       b[lane].r[0], b[lane].r[1]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[4 * lane + i] = c[i];
}

// ---- CPU reference ------------------------------------------------------------

// D[r][n] = sum_k A[r][k] * B[k][n]; exact for integer inputs.
std::vector<float> ref_dot(const std::vector<long long>& A, const std::vector<long long>& B, int K) {
    std::vector<float> D(16 * 8);
    for (int r = 0; r < 16; ++r) {
        for (int n = 0; n < 8; ++n) {
            long long acc = 0;
            for (int k = 0; k < K; ++k) acc += A[static_cast<std::size_t>(r) * K + k] * B[static_cast<std::size_t>(k) * 8 + n];
            D[static_cast<std::size_t>(r) * 8 + n] = static_cast<float>(acc);
        }
    }
    return D;
}

std::vector<int> ref_dot_i32(const std::vector<long long>& A, const std::vector<long long>& B, int K) {
    std::vector<int> D(16 * 8);
    for (int r = 0; r < 16; ++r) {
        for (int n = 0; n < 8; ++n) {
            long long acc = 0;
            for (int k = 0; k < K; ++k) acc += A[static_cast<std::size_t>(r) * K + k] * B[static_cast<std::size_t>(k) * 8 + n];
            D[static_cast<std::size_t>(r) * 8 + n] = static_cast<int>(acc);
        }
    }
    return D;
}

// Per-lane expected c0..c3: c0=(r,2t) c1=(r,2t+1) c2=(r+8,2t) c3=(r+8,2t+1).
template <typename T>
std::vector<T> expected_for_lanes(const std::vector<T>& D) {
    std::vector<T> e(32 * 4);
    for (int lane = 0; lane < 32; ++lane) {
        const int r = frag_r(lane);
        const int t = frag_t(lane);
        e[4 * lane + 0] = D[static_cast<std::size_t>(r) * 8 + 2 * t];
        e[4 * lane + 1] = D[static_cast<std::size_t>(r) * 8 + 2 * t + 1];
        e[4 * lane + 2] = D[static_cast<std::size_t>(r + 8) * 8 + 2 * t];
        e[4 * lane + 3] = D[static_cast<std::size_t>(r + 8) * 8 + 2 * t + 1];
    }
    return e;
}

std::vector<long long> rand_ints(std::size_t n, std::uint32_t seed, int lo) {
    std::vector<float> v(n);
    fill_uniform(v, seed, -static_cast<float>(lo), static_cast<float>(lo));
    std::vector<long long> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = std::llround(v[i]);
    return out;
}

std::vector<long long> ones(std::size_t n) {
    std::vector<long long> v(n, 1);
    return v;
}

// ---- runners ------------------------------------------------------------------

template <typename Kernel, typename OutT>
int run_shape(const char* name, Kernel kernel, const Frag4* a, const Frag2* b, const std::vector<OutT>& expected) {
    Frag4* da = nullptr;
    Frag2* db = nullptr;
    OutT* dout = nullptr;
    cuda_check(cudaMalloc(&da, 32 * sizeof(Frag4)), "malloc da");
    cuda_check(cudaMalloc(&db, 32 * sizeof(Frag2)), "malloc db");
    cuda_check(cudaMalloc(&dout, 128 * sizeof(OutT)), "malloc dout");
    cuda_check(cudaMemcpy(da, a, 32 * sizeof(Frag4), cudaMemcpyHostToDevice), "copy a");
    cuda_check(cudaMemcpy(db, b, 32 * sizeof(Frag2), cudaMemcpyHostToDevice), "copy b");
    cuda_check(cudaMemset(dout, 0xff, 128 * sizeof(OutT)), "memset dout");
    kernel<<<1, 32>>>(da, db, dout);
    cuda_check(cudaDeviceSynchronize(), "sync");
    int failures = 0;
    std::vector<OutT> got(128);
    cuda_check(cudaMemcpy(got.data(), dout, 128 * sizeof(OutT), cudaMemcpyDeviceToHost), "copy out");
    for (int i = 0; i < 128; ++i) {
        if (got[i] != expected[i]) {
            if (failures < 8) {
                std::cout << "  " << name << " lane-mismatch at out[" << i << "]: got " << static_cast<long long>(got[i])
                          << " expected " << static_cast<long long>(expected[i]) << "\n";
            }
            ++failures;
        }
    }
    if (failures == 0) std::cout << "  " << name << ": PASS (128/128 bit-exact)\n";
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    return failures;
}

int run_bf16(std::uint32_t seed, bool use_ones) {
    std::vector<long long> A = use_ones ? ones(16 * 16) : rand_ints(16 * 16, seed, 32);
    std::vector<long long> B = use_ones ? ones(16 * 8) : rand_ints(16 * 8, seed + 1, 32);
    Frag4 a[32];
    Frag2 b[32];
    for (int lane = 0; lane < 32; ++lane) {
        const int r = frag_r(lane);
        const int t = frag_t(lane);
        a[lane].r[0] = pack2(f32_to_bf16(float(A[16 * r + 2 * t])), f32_to_bf16(float(A[16 * r + 2 * t + 1])));
        a[lane].r[1] = pack2(f32_to_bf16(float(A[16 * (r + 8) + 2 * t])),
                             f32_to_bf16(float(A[16 * (r + 8) + 2 * t + 1])));
        a[lane].r[2] = pack2(f32_to_bf16(float(A[16 * r + 2 * t + 8])),
                             f32_to_bf16(float(A[16 * r + 2 * t + 9])));
        a[lane].r[3] = pack2(f32_to_bf16(float(A[16 * (r + 8) + 2 * t + 8])),
                             f32_to_bf16(float(A[16 * (r + 8) + 2 * t + 9])));
        b[lane].r[0] = pack2(f32_to_bf16(float(B[8 * (2 * t) + r])), f32_to_bf16(float(B[8 * (2 * t + 1) + r])));
        b[lane].r[1] = pack2(f32_to_bf16(float(B[8 * (2 * t + 8) + r])),
                             f32_to_bf16(float(B[8 * (2 * t + 9) + r])));
    }
    const std::vector<float> ref = ref_dot(A, B, 16);
    return run_shape(use_ones ? "mma_bf16 ones" : "mma_bf16 random", kernel_bf16, a, b,
                     expected_for_lanes(ref));
}

int run_f16(std::uint32_t seed, bool use_ones) {
    std::vector<long long> A = use_ones ? ones(16 * 16) : rand_ints(16 * 16, seed, 32);
    std::vector<long long> B = use_ones ? ones(16 * 8) : rand_ints(16 * 8, seed + 1, 32);
    Frag4 a[32];
    Frag2 b[32];
    for (int lane = 0; lane < 32; ++lane) {
        const int r = frag_r(lane);
        const int t = frag_t(lane);
        a[lane].r[0] = pack2(enc_f16(A[16 * r + 2 * t]), enc_f16(A[16 * r + 2 * t + 1]));
        a[lane].r[1] = pack2(enc_f16(A[16 * (r + 8) + 2 * t]), enc_f16(A[16 * (r + 8) + 2 * t + 1]));
        a[lane].r[2] = pack2(enc_f16(A[16 * r + 2 * t + 8]), enc_f16(A[16 * r + 2 * t + 9]));
        a[lane].r[3] = pack2(enc_f16(A[16 * (r + 8) + 2 * t + 8]), enc_f16(A[16 * (r + 8) + 2 * t + 9]));
        b[lane].r[0] = pack2(enc_f16(B[8 * (2 * t) + r]), enc_f16(B[8 * (2 * t + 1) + r]));
        b[lane].r[1] = pack2(enc_f16(B[8 * (2 * t + 8) + r]), enc_f16(B[8 * (2 * t + 9) + r]));
    }
    const std::vector<float> ref = ref_dot(A, B, 16);
    return run_shape(use_ones ? "mma_f16 ones" : "mma_f16 random", kernel_f16, a, b,
                     expected_for_lanes(ref));
}

int run_s8(std::uint32_t seed, bool use_ones) {
    std::vector<long long> A = use_ones ? ones(16 * 32) : rand_ints(16 * 32, seed, 127);
    std::vector<long long> B = use_ones ? ones(32 * 8) : rand_ints(32 * 8, seed + 1, 127);
    Frag4 a[32];
    Frag2 b[32];
    for (int lane = 0; lane < 32; ++lane) {
        const int r = frag_r(lane);
        const int t = frag_t(lane);
        a[lane].r[0] = pack4_i8(std::int8_t(A[32 * r + 4 * t]), std::int8_t(A[32 * r + 4 * t + 1]),
                                std::int8_t(A[32 * r + 4 * t + 2]), std::int8_t(A[32 * r + 4 * t + 3]));
        a[lane].r[1] = pack4_i8(std::int8_t(A[32 * (r + 8) + 4 * t]), std::int8_t(A[32 * (r + 8) + 4 * t + 1]),
                                std::int8_t(A[32 * (r + 8) + 4 * t + 2]), std::int8_t(A[32 * (r + 8) + 4 * t + 3]));
        a[lane].r[2] = pack4_i8(std::int8_t(A[32 * r + 16 + 4 * t]), std::int8_t(A[32 * r + 16 + 4 * t + 1]),
                                std::int8_t(A[32 * r + 16 + 4 * t + 2]), std::int8_t(A[32 * r + 16 + 4 * t + 3]));
        a[lane].r[3] = pack4_i8(std::int8_t(A[32 * (r + 8) + 16 + 4 * t]),
                                std::int8_t(A[32 * (r + 8) + 16 + 4 * t + 1]),
                                std::int8_t(A[32 * (r + 8) + 16 + 4 * t + 2]),
                                std::int8_t(A[32 * (r + 8) + 16 + 4 * t + 3]));
        b[lane].r[0] = pack4_i8(std::int8_t(B[8 * (4 * t) + r]), std::int8_t(B[8 * (4 * t + 1) + r]),
                                std::int8_t(B[8 * (4 * t + 2) + r]), std::int8_t(B[8 * (4 * t + 3) + r]));
        b[lane].r[1] = pack4_i8(std::int8_t(B[8 * (16 + 4 * t) + r]), std::int8_t(B[8 * (16 + 4 * t + 1) + r]),
                                std::int8_t(B[8 * (16 + 4 * t + 2) + r]), std::int8_t(B[8 * (16 + 4 * t + 3) + r]));
    }
    const std::vector<int> ref = ref_dot_i32(A, B, 32);
    return run_shape(use_ones ? "mma_s8 ones" : "mma_s8 random", kernel_s8, a, b,
                     expected_for_lanes(ref));
}

int run_tf32(std::uint32_t seed, bool use_ones) {
    std::vector<long long> A = use_ones ? ones(16 * 8) : rand_ints(16 * 8, seed, 32);
    std::vector<long long> B = use_ones ? ones(8 * 8) : rand_ints(8 * 8, seed + 1, 32);
    Frag4 a[32];
    Frag2 b[32];
    for (int lane = 0; lane < 32; ++lane) {
        const int r = frag_r(lane);
        const int t = frag_t(lane);
        a[lane].r[0] = enc_tf32_bits(A[8 * r + t]);
        a[lane].r[1] = enc_tf32_bits(A[8 * (r + 8) + t]);
        a[lane].r[2] = enc_tf32_bits(A[8 * r + t + 4]);
        a[lane].r[3] = enc_tf32_bits(A[8 * (r + 8) + t + 4]);
        b[lane].r[0] = enc_tf32_bits(B[8 * t + r]);
        b[lane].r[1] = enc_tf32_bits(B[8 * (t + 4) + r]);
    }
    const std::vector<float> ref = ref_dot(A, B, 8);
    return run_shape(use_ones ? "mma_tf32 ones" : "mma_tf32 random", kernel_tf32, a, b,
                     expected_for_lanes(ref));
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (std::uint32_t seed : {1u, 2u, 3u}) {
        failures += run_bf16(seed, false);
        failures += run_f16(seed, false);
        failures += run_s8(seed, false);
        failures += run_tf32(seed, false);
    }
    failures += run_bf16(99u, true);
    failures += run_f16(99u, true);
    failures += run_s8(99u, true);
    failures += run_tf32(99u, true);
    if (failures) {
        std::cout << "FAIL: " << failures << " mismatches\n";
        return 1;
    }
    std::cout << "PASS: all mma fragment shapes bit-exact\n";
    return 0;
}
