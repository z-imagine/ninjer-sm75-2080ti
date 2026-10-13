#pragma once

#include "ninfer/ops/context_kv_materialize.h"

namespace ninfer::ops::detail {

enum class ContextKVMaterializeRoute : std::uint8_t {
    KSplit16,
    KSplit24,
    Mma32,
    Mma80,
    Mma96,
    Fused64,
    Mma64,
};

// Columns count the envelope's live prefix capacity across requests, not physical padding.
constexpr ContextKVMaterializeRoute context_kv_materialize_route(std::int32_t columns) {
    using Route = ContextKVMaterializeRoute;
    if (columns <= 16) return Route::KSplit16;
    if (columns <= 24) return Route::KSplit24;
    if (columns <= 32) return Route::KSplit16;
    if (columns <= 64) return Route::Mma32;
#if defined(NINFER_SM75)
    // The 2080Ti opt-in shared-memory cap is 64 KiB, but the Mma80 (66048 B),
    // Mma96 (74240 B), and Fused64 (99328 B) mma tiles exceed it, and none can
    // be shrunk into it: the 5120-wide K forces BlockK to a multiple of 32 with
    // 4/8/16 B cp transfers. Fall back to the Mma64 tile (37120 B), which uses
    // the scratch + post-kernel path and yields identical results for any
    // column count.
    return Route::Mma64;
#else
    if (columns <= 80) return Route::Mma80;
    if (columns <= 96) return Route::Mma96;
    if (columns <= 256) return Route::Fused64;
    return Route::Mma64;
#endif
}

constexpr bool context_kv_materialize_uses_scratch(ContextKVMaterializeRoute route) {
    return route != ContextKVMaterializeRoute::Fused64;
}

void context_kv_materialize_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, ContextKVMaterializeRoute route,
    const Tensor& key_scratch, cudaStream_t stream);

} // namespace ninfer::ops::detail
