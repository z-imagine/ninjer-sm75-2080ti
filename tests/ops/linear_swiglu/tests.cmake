add_library(ninfer_linear_swiglu_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_swiglu_test_common.cpp")
ninfer_test_includes(ninfer_linear_swiglu_test_support)
ninfer_op_oracle_options(ninfer_linear_swiglu_test_support)
target_link_libraries(ninfer_linear_swiglu_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_swiglu_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

# NVFP4 / FP8 linear_swiglu is not supported on sm_75; dispatchers throw
# explicit errors, so run these variants only on architectures with the formats.
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
  ninfer_add_op_test(ninfer_linear_swiglu_nvfp4_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4.cpp"
    LIBRARIES ninfer_linear_swiglu_test_support)

  ninfer_add_op_test(ninfer_linear_swiglu_fp8_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
    LIBRARIES ninfer_linear_swiglu_test_support)
endif()
