#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n4096_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    // sm75 p7: multi-row multi-token GEMV (4 rows x 8 warps, __ldcs, x reused across rows):
    // 34816x5120 T=4 measures 399-406 us vs 1257.7 us for the T=1 x4 batched (3.1x).
    if (tokens == 2) return launch_q4_a16_gemv_r4_w8_t2;
    if (tokens == 3) return launch_q4_a16_gemv_r4_w8_t3;
    if (tokens <= 4) return launch_q4_a16_gemv_r4_w8_t4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 320) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
