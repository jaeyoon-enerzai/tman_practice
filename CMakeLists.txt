cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -- You can select to compile AOT or RUNTIME
option(BUILD_AOT "Build x86 64  host offline compiler" ON)
option(BUILD_RUNTIME "Build Android aarch64 runtime executable" OFF)

if(NOT DEFINED QNN_SDK_ROOT OR QNN_SDK_ROOT STREQUAL "")
  message(FATAL_ERROR "Please set -DQNN_SDK_ROOT=/path/to/qnn_sdk (or export QNN_SDK_ROOT and pass it).")
endif()

message(STATUS "QNN_SDK_ROOT = ${QNN_SDK_ROOT}")

# Many SDKs provide libs under something like:
#   ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/   or /lib/x86_64-linux-gcc/
# You can pass QNN_LIB_DIR explicitly if you want.
if(NOT DEFINED QNN_LIB_DIR)
  # Best-effort default (edit this to match your SDK)
  set(QNN_LIB_DIR "${QNN_SDK_ROOT}/lib/x86_64-linux-clang")
endif()


# ---- Paths (QNN SDK layout can vary by version; adjust if needed) ----
set(QNN_INC_DIR "${QNN_SDK_ROOT}/include/QNN")
message(STATUS "QNN_INC_DIR  = ${QNN_INC_DIR}")


# common library
add_subdirectory(common)

if(BUILD_AOT)
  add_subdirectory(aot)
endif()

if(BUILD_RUNTIME)
  add_subdirectory(runtime)
endif()