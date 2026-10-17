#include "ops/linear/q4/q4_instance_launch.cuh"

namespace ninfer::ops::detail {

void launch_q4_a16_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q4_a16_gemv_instance<q4_instances::GemvR4W1>(x, w, out, stream);
}

void launch_q4_a16_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                     cudaStream_t stream) {
    launch_q4_a16_gemv_instance<q4_instances::GemvR1W8K5120>(x, w, out, stream);
}

void launch_q4_a16_gemv_r1_w8_k6144(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    launch_q4_a16_gemv_instance<q4_instances::GemvR1W8K6144>(x, w, out, stream);
}

namespace {
// sm75 (2080Ti): the T=2..4 sliced-K MMA instances for small-N shapes run at 9-55% DRAM and
// cost 2.7-5.4x the T=1 GEMV per token (NCU: register-bound or L1-bound, occupancy 25-37%).
// Re-issue the T=1 GEMV per token: activations are [K, T] column-major, token t at t*K.
template <auto Launch>
void launch_q4_gemv_batched(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::size_t x_stride = std::size_t{w.k} * sizeof(__nv_bfloat16);
    const std::size_t o_stride = std::size_t{w.n} * sizeof(__nv_bfloat16);
    for (std::int32_t t = 0; t < tokens; ++t) {
        Tensor xt = x;
        xt.data   = static_cast<char*>(x.data) + std::size_t{t} * x_stride;
        xt.ne[1]  = 1;
        Tensor ot = out;
        ot.data   = static_cast<char*>(out.data) + std::size_t{t} * o_stride;
        Launch(xt, w, ot, stream);
    }
}
} // namespace

void launch_q4_a16_gemv_r1_w8_direct_batched(const Tensor& x, const Weight& weight, Tensor& out,
                                             cudaStream_t stream) {
    launch_q4_gemv_batched<&launch_q4_a16_gemv_r1_w8_direct>(x, weight, out, stream);
}

void launch_q4_a16_gemv_r1_w8_k6144_batched(const Tensor& x, const Weight& weight, Tensor& out,
                                            cudaStream_t stream) {
    launch_q4_gemv_batched<&launch_q4_a16_gemv_r1_w8_k6144>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
