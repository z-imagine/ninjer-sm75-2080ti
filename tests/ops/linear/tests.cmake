add_library(ninfer_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninfer_test_includes(ninfer_linear_test_support)
ninfer_op_oracle_options(ninfer_linear_test_support)
target_link_libraries(ninfer_linear_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

# NVFP4 / FP8 linear GEMMs are not supported on sm_75 (no FP8/NVFP4 tensor
# cores); the dispatchers throw explicit errors, so run these variants only on
# architectures that actually carry the formats.
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
  ninfer_add_op_test(ninfer_linear_nvfp4_a16_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
    LIBRARIES ninfer_linear_test_support)

  ninfer_add_op_test(ninfer_linear_nvfp4_a4_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
    LIBRARIES ninfer_linear_test_support)

  ninfer_add_op_test(ninfer_linear_fp8_a16_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
    LIBRARIES ninfer_linear_test_support)

  ninfer_add_op_test(ninfer_linear_fp8_a8_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
    LIBRARIES ninfer_linear_test_support)
endif()

ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_linear_test_support)
