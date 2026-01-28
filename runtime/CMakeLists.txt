if(NOT ANDROID)
  message(FATAL_ERROR "runtime target must be built with Android toolchain (NDK).")
endif()

if(NOT DEFINED QNN_ANDROID_LIB_DIR OR QNN_ANDROID_LIB_DIR STREQUAL "")
  set(QNN_ANDROID_LIB_DIR "${QNN_SDK_ROOT}/lib/aarch64-android")
endif()

add_executable(qnn_runtime_runner
  src/main_run.cpp
)

target_include_directories(qnn_runtime_runner PRIVATE
  ${QNN_INC_DIR}
  ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(qnn_runtime_runner PRIVATE dl qnn_common)

target_link_libraries(qnn_runtime_runner PRIVATE
  "${QNN_ANDROID_LIB_DIR}/libQnnSystem.so"
  "${QNN_ANDROID_LIB_DIR}/libQnnHtp.so"
)
