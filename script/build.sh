#!/bin/bash

MODE=$1
if [ "$MODE" = "aot" ]; then
cd ..
mkdir -p build_aot
cd build_aot

cmake .. -DQNN_SDK_ROOT=/workspace/2.37.1.250807 \
        -DQNN_LIB_DIR=/workspace/2.37.1.250807/lib/x86_64-linux-clang \
        -DBUILD_AOT=ON -DBUILD_RUNTIME=OFF -DCMAKE_CXX_COMPILER=clang++

make -j
else
cd ..
mkdir -p build_runtime
cd build_runtime

cmake .. -DBUILD_AOT=OFF -DBUILD_RUNTIME=ON \
        -DQNN_SDK_ROOT=/workspace/2.37.1.250807 \
        -DCMAKE_TOOLCHAIN_FILE=/workspace/android-ndk-r26c/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static
make -j
fi