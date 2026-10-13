#pragma once

#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct Int8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles  = (Tokens * G::GroupSize + 15) / 16;
#if defined(NINFER_SM75)
    // The 2080Ti opt-in shared-memory cap is 64 KiB: the 64-row key-tile arena
    // alone is 4*64*256 = 64 KiB of dynamic memory, and with the kernel's static
    // shared memory it exceeds the cap (cudaFuncSetAttribute rejects it). The
    // 32-row key tile (32-KiB arena) fits and yields identical results.
    static constexpr bool kWideKeys = false;
#else
    static constexpr bool kWideKeys = G::GroupSize == 8 && kRowTiles >= 3;
#endif
    using Schedule                  = Int8KvGroupedMmaSchedule<Tokens,
                                              kWideKeys        ? 4 * kRowTiles
                                                               : kRowTiles <= 2 ? 8
                                                                                : 2 * kRowTiles,
                                              kWideKeys ? 64 : 32, kWideKeys ? 1 : 2>;
    using Merge                     = Int8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

#if defined(NINFER_SM75)
// 2080Ti opt-in shared-memory cap is 64 KiB: 32-row key tiles keep this at 54 KiB.
using Int8KvTiledInstance = Int8KvTiledMmaSchedule<64, 32, 128>;
#else
using Int8KvTiledInstance = Int8KvTiledMmaSchedule<64, 64, 128>;
#endif

} // namespace ninfer::ops::detail
