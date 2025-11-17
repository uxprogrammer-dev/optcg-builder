#!/bin/bash
# GPU diagnostic script for RunPod

echo "=== GPU Diagnostic Check ==="
echo ""

echo "1. Checking nvidia-smi (NVIDIA driver)..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "❌ nvidia-smi not found - NVIDIA drivers may not be installed"
fi
echo ""

echo "2. Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "⚠️  nvcc not found in PATH (may still be installed)"
fi
echo ""

echo "3. Checking CUDA library paths..."
if [ -d "/usr/local/cuda" ]; then
    echo "✓ /usr/local/cuda exists"
    ls -la /usr/local/cuda*/bin/nvcc 2>/dev/null || echo "  No nvcc found in /usr/local/cuda"
else
    echo "❌ /usr/local/cuda not found"
fi
echo ""

echo "4. Checking LD_LIBRARY_PATH..."
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

echo "5. Checking for CUDA libraries..."
for lib in libcudart.so libcublas.so libcudnn.so; do
    if find /usr -name "$lib*" 2>/dev/null | head -1 | grep -q .; then
        echo "✓ Found $lib"
    else
        echo "⚠️  $lib not found in /usr"
    fi
done
echo ""

echo "6. Checking TensorFlow CUDA libraries..."
VENV_LIB="$PWD/ml/.venv/lib/python3.12/site-packages"
if [ -d "$VENV_LIB/nvidia" ]; then
    echo "✓ TensorFlow NVIDIA libraries found:"
    find "$VENV_LIB/nvidia" -name "*.so" | head -5
else
    echo "❌ TensorFlow NVIDIA libraries not found"
fi
echo ""

echo "7. Testing TensorFlow GPU detection..."
python3 << 'EOF'
import sys
import os
sys.path.insert(0, '/workspace/optcg-builder')

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA built: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo "=== Diagnostic Complete ==="

