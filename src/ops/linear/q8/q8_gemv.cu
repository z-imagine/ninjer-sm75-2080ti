#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"

namespace ninfer::ops::detail {
void launch_q8_a16_gemv_r4_w1_k16384(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q8_a16_gemv<Q8A16GemvSchedule<4, 1, 2, 16384>>(
        q8_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void launch_q8_a16_gemv_r4_w1_k16384_batched(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream) {
    // sm75 (2080Ti): the T=2..4 sliced-K MMA path for this shape runs at ~25% occupancy
    // (NCU: 255 registers/thread, one block per SM) and at ~12x the DRAM floor (T=2: 776 us
    // vs the 67 us T=1 GEMV at the same floor). Re-issuing the T=1 GEMV per token trades
    // T-fold weight traffic for full memory-level parallelism; activations are [K, T]
    // column-major, so token t lives at offset t*K.
    const std::int32_t tokens = x.ne[1];
    const std::size_t x_stride = std::size_t{w.k} * sizeof(__nv_bfloat16);
    const std::size_t o_stride = std::size_t{w.n} * sizeof(__nv_bfloat16);
    for (std::int32_t t = 0; t < tokens; ++t) {
        Tensor xt = x;
        xt.data   = static_cast<char*>(x.data) + std::size_t{t} * x_stride;
        xt.ne[1]  = 1;
        Tensor ot = out;
        ot.data   = static_cast<char*>(out.data) + std::size_t{t} * o_stride;
        launch_q8_a16_gemv_r4_w1_k16384(xt, w, ot, stream);
    }
}
} // namespace ninfer::ops::detail
