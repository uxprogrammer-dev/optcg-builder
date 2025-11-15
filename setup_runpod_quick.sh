#!/bin/bash
# Quick RunPod setup script - optimized for fast reconfiguration
# Run this script on a fresh RunPod instance

set -e

echo "=== Quick RunPod Setup (Fast Reconfiguration) ==="
echo ""

# Step 1: Clone repository
echo "Step 1: Cloning repository..."
cd /workspace
if [ ! -d "optcg-builder" ]; then
    git clone https://github.com/uxprogrammer-dev/optcg-builder.git
fi
cd optcg-builder
git pull  # Update if already exists
echo "✓ Repository ready"
echo ""

# Step 2: Set up Python virtual environment
echo "Step 2: Setting up Python virtual environment..."
if [ ! -d "ml/.venv" ]; then
    python3 -m venv ml/.venv
fi
source ml/.venv/bin/activate
pip install --upgrade pip --quiet
echo "✓ Virtual environment ready"
echo ""

# Step 3: Install dependencies (only if needed)
echo "Step 3: Checking dependencies..."
if ! python -c "import tensorflow" 2>/dev/null; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install -r ml/requirements.txt --quiet
else
    echo "Dependencies already installed, skipping..."
fi
echo "✓ Dependencies ready"
echo ""

# Step 4: Set up GPU environment
echo "Step 4: Configuring GPU environment..."
export PYTHONPATH=/workspace/optcg-builder:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

VENV_LIB=/workspace/optcg-builder/ml/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$VENV_LIB/nvidia/cufft/lib:$VENV_LIB/nvidia/curand/lib:$VENV_LIB/nvidia/cusolver/lib:$VENV_LIB/nvidia/cusparse/lib:$VENV_LIB/nvidia/cuda_runtime/lib:$VENV_LIB/nvidia/nccl/lib:$LD_LIBRARY_PATH

echo "✓ GPU environment configured"
echo ""

# Step 5: Verify GPU detection
echo "Step 5: Verifying GPU detection..."
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('GPUs detected:', len(gpus)); [print(f'  GPU {i}: {gpu}') for i, gpu in enumerate(gpus)]"
echo ""

# Step 6: Display ready message
echo "=== Setup Complete! Ready for Training ==="
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

