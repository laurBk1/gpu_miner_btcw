#!/bin/bash

# Build script for BTCW OpenCL Miner

echo "Building BTCW OpenCL Miner for AMD GPUs..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Executable: build/gpu_miner_opencl"
    echo ""
    echo "Usage:"
    echo "  ./gpu_miner_opencl [gpu_number]"
    echo ""
    echo "Example:"
    echo "  ./gpu_miner_opencl 1"
else
    echo "Build failed!"
    exit 1
fi