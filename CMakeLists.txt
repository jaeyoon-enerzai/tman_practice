# This code builds a script to compile QNN graph by offline
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED QNN_SDK_ROOT OR QNN_SDK_ROOT STREQUAL "")
  message(FATAL_ERROR "Please set -DQNN_SDK_ROOT=/path/to/qnn_sdk (or export QNN_SDK_ROOT and pass it).")
endif()

message(STATUS "QNN_SDK_ROOT = ${QNN_SDK_ROOT}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  # strip symbols
  add_link_options("-s")

  # --gc-sections is added by torch.
  add_compile_options("-O3" "-ffunction-sections" "-fdata-sections" "-frtti")
endif()

# ---- Paths (QNN SDK layout can vary by version; adjust if needed) ----
set(QNN_INC_DIR "${QNN_SDK_ROOT}/include/QNN")

# Many SDKs provide libs under something like:
#   ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/   or /lib/x86_64-linux-gcc/
# You can pass QNN_LIB_DIR explicitly if you want.
if(NOT DEFINED QNN_LIB_DIR)
  # Best-effort default (edit this to match your SDK)
  set(QNN_LIB_DIR "${QNN_SDK_ROOT}/lib/x86_64-linux-clang")
endif()

message(STATUS "QNN_SDK_ROOT = ${QNN_SDK_ROOT}")
message(STATUS "QNN_INC_DIR  = ${QNN_INC_DIR}")
message(STATUS "QNN_LIB_DIR  = ${QNN_LIB_DIR}")

# ---- Target ----
add_executable(qnn_offline_compiler
  src/main.cpp
  src/qnn_dynload.cpp
  src/qnn_backend.cpp
  src/qnn_device.cpp
  src/qnn_context.cpp
  src/qnn_graph.cpp
  src/qnn_tensor.cpp
  src/qnn_platform.cpp
  src/qnn_graph_config.cpp
)

target_include_directories(qnn_offline_compiler PRIVATE
  ${QNN_INC_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(CMAKE_CXX_COMPILER clang++)
add_compile_options(-stdlib=libc++)
add_link_options(-stdlib=libc++)

# If you use dlopen/dlsym yourself you need dl; if not, you may not.
# But many projects still link it for safety.
target_link_libraries(qnn_offline_compiler PRIVATE c++ c++abi pthread dl)

# ---- Link QNN libs directly (no manual dlopen in your code) ----
# The exact library names may differ by SDK version.
# Common ones:
#   libQnnHtp.so, libQnnCpu.so, libQnnSystem.so
#   plus op packages sometimes: libQnnHtpOpPackage.so (name varies)
#
# For OFFLINE COMPILE targeting HTP, typically you need:
#   - QnnSystem (for binary info / system context utilities, sometimes optional)
#   - QnnHtp backend library (for graph construction + offline compilation)
#
# If your SDK ships import targets / cmake config, you could use find_package,
# but many don't. So we link by absolute path.
target_link_libraries(qnn_offline_compiler PRIVATE
  "${QNN_LIB_DIR}/libQnnSystem.so"
  "${QNN_LIB_DIR}/libQnnHtp.so"
)

# Optional: if you plan to also validate on host CPU backend
# target_link_libraries(qnn_offline_compiler PRIVATE
#   "${QNN_LIB_DIR}/libQnnCpu.so"
# )

# ---- RPATH so you can run without setting LD_LIBRARY_PATH every time ----
if(QNN_ENABLE_RPATH)
  set_target_properties(qnn_offline_compiler PROPERTIES
    BUILD_RPATH "${QNN_LIB_DIR}"
    INSTALL_RPATH "${QNN_LIB_DIR}"
  )
endif()