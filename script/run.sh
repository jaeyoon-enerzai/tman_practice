#!/bin/bash

# QNN_SDK=/workspace/2.37.1.250807/
QNN_SDK=/workspace/2.40.0.251030/
WORKSP=/data/local/tmp/htprun
# adb push $QNN_SDK/lib/aarch64-android/libQnnHtp.so $WORKSP
# adb push $QNN_SDK/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so $WORKSP
# adb push $QNN_SDK/lib/aarch64-android/libQnnHtpV73Stub.so $WORKSP
# adb push $QNN_SDK/lib/aarch64-android/libQnnHtpPrepare.so $WORKSP
# adb push $QNN_SDK/lib/aarch64-android/libQnnSystem.so $WORKSP
# adb push $QNN_SDK/lib/aarch64-android/libQnnModelDlc.so $WORKSP

adb push ../build_runtime/runtime/qnn_runtime_runner  $WORKSP
adb push add_graph.bin $WORKSP
adb push static_q.bin $WORKSP
adb push static_k.bin $WORKSP
adb push static_v.bin $WORKSP

adb shell "cd /data/local/tmp/htprun && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD ./qnn_runtime_runner"