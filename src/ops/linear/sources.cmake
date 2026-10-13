target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/linear.cpp"
)

include("${CMAKE_CURRENT_LIST_DIR}/bf16/sources.cmake")
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
  # SM75 has no FP8/NVFP4 tensor cores; the families are excluded below.
  include("${CMAKE_CURRENT_LIST_DIR}/fp8/sources.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/nvfp4/sources.cmake")
else()
  # Pure-host FP8 geometry validation stays compiled so wrappers can validate layouts.
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_format.cpp")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/q4/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q5/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q6/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q8/sources.cmake")
