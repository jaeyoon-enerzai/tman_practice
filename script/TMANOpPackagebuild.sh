#!/bin/bash
cd /workspace/Hexagon_SDK/6.4.0.2 
source setup_sdk_env.source
cd /workspace/2.40.0.251030/bin
source envsetup.sh
cd /workspace/TMANOpPackage
rm -rf build
make htp_x86 htp_v73
export ADB_SERVER_SOCKET=tcp:127.0.0.1:15037
cd build/hexagon-v73
adb -s b4a7bb34 push TMANOpPackageInterface.d /data/local/tmp/llama/
adb -s b4a7bb34 push TMANOpPackageInterface.o /data/local/tmp/llama/
adb -s b4a7bb34 push fp_extend.d /data/local/tmp/llama/
adb -s b4a7bb34 push fp_extend.o /data/local/tmp/llama/
adb -s b4a7bb34 push fp_trunc.d /data/local/tmp/llama/
adb -s b4a7bb34 push fp_trunc.o /data/local/tmp/llama/
adb -s b4a7bb34 push libQnnTMANOpPackage.so /data/local/tmp/llama/
adb -s b4a7bb34 push ops /data/local/tmp/llama/
