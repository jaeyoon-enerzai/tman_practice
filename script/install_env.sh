#!/bin/bash

apt-get -y update
apt-get install python3.10 python3.10-dev

apt-get install -y build-essential ninja-build git pkg-config \
  gdb lldb unzip zip curl rsync \
  software-properties-common

add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get install -y g++-13 gcc-13

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

apt-get install -y ca-certificates gpg wget

cd /opt
curl -fL -o cmake-3.24.4-linux-x86_64.tar.gz \
  https://github.com/Kitware/CMake/releases/download/v3.24.4/cmake-3.24.4-linux-x86_64.tar.gz

tar -xzf cmake-3.24.4-linux-x86_64.tar.gz
ln -sf /opt/cmake-3.24.4-linux-x86_64/bin/cmake /usr/local/bin/cmake
hash -r

apt-get install -y libc++1 libc++abi1

apt-get install -y clang lld llvm

apt-get install -y build-essential g++ \
  libc++-dev libc++abi-dev \
  clang

# pip install torch numpy==1.24