#!/bin/bash
# Complete setup script for training on RunPod RTX 4090
# Run this script on your RunPod instance

set -e  # Exit on error

echo "=== RunPod RTX 4090 Setup Script ==="
echo ""

# Step 1: Clone repository
echo "Step 1: Cloning repository..."
cd /workspace
if [ ! -d "optcg-builder" ]; then
    git clone git@github.com:uxprogrammer-dev/optcg-builder.git || git clone https://github.com/uxprogrammer-dev/optcg-builder.git
fi
cd optcg-builder
echo "✓ Repository cloned"
echo ""

# Step 2: Set up Python virtual environment
echo "Step 2: Setting up Python virtual environment..."
python3 -m venv ml/.venv
source ml/.venv/bin/activate
pip install --upgrade pip
echo "✓ Virtual environment created"
echo ""

# Step 3: Install dependencies
echo "Step 3: Installing Python dependencies (this may take a few minutes)..."
pip install -r ml/requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 4: Set up GPU environment
echo "Step 4: Configuring GPU environment..."
export PYTHONPATH=/workspace/optcg-builder:$PYTHONPATH

# Set CUDA library paths
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Add TensorFlow's bundled NVIDIA libraries
VENV_LIB=/workspace/optcg-builder/ml/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$VENV_LIB/nvidia/cufft/lib:$VENV_LIB/nvidia/curand/lib:$VENV_LIB/nvidia/cusolver/lib:$VENV_LIB/nvidia/cusparse/lib:$VENV_LIB/nvidia/cuda_runtime/lib:$VENV_LIB/nvidia/nccl/lib:$LD_LIBRARY_PATH

echo "✓ GPU environment configured"
echo ""

# Step 5: Verify GPU detection
echo "Step 5: Verifying GPU detection..."
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs detected:', len(gpus)); [print(f'  GPU {i}: {gpu}') for i, gpu in enumerate(gpus)]"
echo ""

# Step 6: Display training command
echo "=== Setup Complete! ==="
echo ""
echo "To start training, run:"
echo ""
echo "cd /workspace/optcg-builder"
echo "source ml/.venv/bin/activate"
echo "export PYTHONPATH=/workspace/optcg-builder:\$PYTHONPATH"
echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH"
echo "VENV_LIB=/workspace/optcg-builder/ml/.venv/lib/python3.12/site-packages"
echo "export LD_LIBRARY_PATH=\$VENV_LIB/nvidia/cudnn/lib:\$VENV_LIB/nvidia/cublas/lib:\$VENV_LIB/nvidia/cufft/lib:\$VENV_LIB/nvidia/curand/lib:\$VENV_LIB/nvidia/cusolver/lib:\$VENV_LIB/nvidia/cusparse/lib:\$VENV_LIB/nvidia/cuda_runtime/lib:\$VENV_LIB/nvidia/nccl/lib:\$LD_LIBRARY_PATH"
echo ""
echo "python -m ml.training.train \\"
echo "  --dataset ml/artifacts/combined_prompt_deck.jsonl \\"
echo "  --output-dir models \\"
echo "  --epochs 40 \\"
echo "  --batch-size 48 \\"
echo "  --learning-rate 2e-4 \\"
echo "  --coverage-weight 0.05"
echo ""

