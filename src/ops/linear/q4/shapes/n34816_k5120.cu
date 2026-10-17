#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n34816_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    // sm75: sliced K=5120 at T=2..4 costs 2.7x the T=1 GEMV per token (NCU: L1-bound, 37%
    // occupancy); batched GEMV measured T=2 1709 -> ~632 us.
    if (tokens <= 4) return launch_q4_a16_gemv_r1_w8_direct_batched;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 63) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 120) return launch_q4_a16_mma_r64_t120;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 240) return launch_q4_a16_mma_r64_t120;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 288) return launch_q4_a16_mma_r64_t96;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
