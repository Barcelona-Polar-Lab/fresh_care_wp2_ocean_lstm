#!/bin/bash

echo "=== GPU Diagnostics ==="
echo ""

echo "1. Checking for NVIDIA GPU..."
lspci | grep -i nvidia
echo ""

echo "2. Checking NVIDIA driver (nvidia-smi)..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found - NVIDIA drivers may not be installed"
fi
echo ""

echo "3. Checking CUDA toolkit..."
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found - CUDA toolkit may not be installed"
fi
echo ""

echo "4. Checking for CUDA libraries..."
ldconfig -p | grep cuda
echo ""

echo "5. Checking environment variables..."
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "6. Checking Python CUDA libraries..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA compiled version: {torch.version.cuda}'); import torch.backends.cudnn as cudnn; print(f'cuDNN available: {cudnn.is_available()}')"
echo ""

echo "7. Checking kernel modules..."
lsmod | grep nvidia
echo ""

echo "8. Checking device files..."
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* devices found"
echo ""
