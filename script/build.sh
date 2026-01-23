#!/bin/bash

cd ..
mkdir -p build
cd build

cmake .. -DQNN_SDK_ROOT=/workspace/2.37.1.250807 \
        -DQNN_LIB_DIR=/workspace/2.37.1.250807/lib/x86_64-linux-clang

make -j