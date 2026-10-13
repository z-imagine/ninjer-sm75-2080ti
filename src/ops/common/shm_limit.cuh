#pragma once

// Per-block shared memory capacity for kernels that opt in via
// cudaFuncAttributeMaxDynamicSharedMemorySize.
//
// Measured on 2080 Ti (sm_75, CUDA 12.4 runtime, device-property probe
// /tmp/sm75_shm_probe.cu): sharedMemPerMultiprocessor = 64 KiB,
// sharedMemPerBlockOptin = 64 KiB. The 99 KiB value used elsewhere is the
// sm86-class limit (100 KiB/SM - 1 KiB); on sm_75 any opt-in request above
// 64 KiB fails with cudaErrorInvalidValue at cudaFuncSetAttribute time.
#if defined(NINFER_SM75)
inline constexpr int NINFER_MAX_BLOCK_SHARED_BYTES = 64 * 1024;
#else
inline constexpr int NINFER_MAX_BLOCK_SHARED_BYTES = 99 * 1024;
#endif
