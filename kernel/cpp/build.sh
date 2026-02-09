#!/bin/bash
# compile.sh

set -e  # Exit on error

echo "=== Compiling HVX Dequantization Test ==="

# Compiler settings
CXX=${CXX:-g++}
CXXFLAGS="-std=c++14 -O3 -march=native -Wall -Wextra"
LDFLAGS=""

# Output binary
OUTPUT="test_hvx_dequantize"

echo "Compiler: $CXX"
echo "Flags: $CXXFLAGS"
echo ""

# Compile
echo "Compiling..."
$CXX $CXXFLAGS test_dequantize_weights.cpp -o $OUTPUT $LDFLAGS

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo "Output binary: $OUTPUT"
    echo ""
    echo "To run the test:"
    echo "  1. Generate test data: python save_test_data.py"
    echo "  2. Run test: ./$OUTPUT"
else
    echo "✗ Compilation failed!"
    exit 1
fi