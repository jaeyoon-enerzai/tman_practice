# ---- common library ----
add_library(qnn_common STATIC
  src/qnn_dynload.cpp
  src/qnn_backend.cpp
  src/qnn_device.cpp
  src/qnn_context.cpp
  src/qnn_graph.cpp
  src/qnn_tensor.cpp
  src/qnn_platform.cpp
  src/qnn_graph_config.cpp
  src/qnn_backendcache.cpp
  src/qnn_mem_manager.cpp
  src/qnn_sharedbuffer.cpp
)
target_include_directories(qnn_common PRIVATE
  ${QNN_INC_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/include
)

# 빌드 타입 옵션(원래 너 설정 유지)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options("-O3" "-ffunction-sections" "-fdata-sections" "-frtti")
  add_link_options("-s")
endif()

# dl은 대부분 필요
if(BUILD_AOT)
  target_compile_options(qnn_common PRIVATE -stdlib=libc++)
  target_link_options(qnn_common PRIVATE -stdlib=libc++)
  target_link_libraries(qnn_common PUBLIC c++ c++abi)
endif()  
target_link_libraries(qnn_common PUBLIC dl)

# Android면 log 추가(필요 없으면 제거 가능)
if(ANDROID)
  target_link_libraries(qnn_common PUBLIC log)
endif()