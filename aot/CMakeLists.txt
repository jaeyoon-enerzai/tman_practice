# ---- Target ----
add_executable(qnn_offline_compiler
  src/main_aot.cpp
)

target_include_directories(qnn_offline_compiler PRIVATE
  ${QNN_INC_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_SOURCE_DIR}/common/include
)

target_link_libraries(qnn_offline_compiler PRIVATE c++ c++abi pthread dl qnn_common)

target_link_libraries(qnn_offline_compiler PRIVATE
  "${QNN_LIB_DIR}/libQnnSystem.so"
  "${QNN_LIB_DIR}/libQnnHtp.so"
)

target_link_options(qnn_offline_compiler PRIVATE -stdlib=libc++)
target_compile_options(qnn_offline_compiler PRIVATE -stdlib=libc++)


if(QNN_ENABLE_RPATH)
  set_target_properties(qnn_offline_compiler PROPERTIES
    BUILD_RPATH "${QNN_LIB_DIR}"
    INSTALL_RPATH "${QNN_LIB_DIR}"
  )
endif()