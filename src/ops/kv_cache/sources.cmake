target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/append/kv_cache_append.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/append/launch.cu"
  # SM75 has no NVFP4 support (V-cache NVFP4 codecs need CUDA 12.8 cuda_fp4.h); excluded below.
  $<$<NOT:$<STREQUAL:${CMAKE_CUDA_ARCHITECTURES},75>>:
    "${CMAKE_CURRENT_LIST_DIR}/append/nvfp4_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/append/k8v4_launch.cu">
)
