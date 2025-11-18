#!/bin/bash
# Complete RunPod setup script for RTX 4090
# Run this script on a fresh RunPod pod

set -e

echo "=== RunPod RTX 4090 Complete Setup ==="
echo ""

# Step 0: Verify GPU access
echo "Step 0: Verifying GPU access..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found, checking GPU..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️  nvidia-smi failed (GPU may not be accessible)"
else
    echo "⚠️  nvidia-smi not found - GPU drivers may not be installed"
    echo "   This is OK if using RunPod's GPU-enabled templates"
fi
echo ""

# Step 1: Clone repository
echo "Step 1: Cloning repository..."
cd /workspace
if [ ! -d "optcg-builder" ]; then
    git clone https://github.com/uxprogrammer-dev/optcg-builder.git
fi
cd optcg-builder
git pull origin main
echo "✓ Repository ready"
echo ""

# Step 2: Set up Python virtual environment
echo "Step 2: Setting up Python virtual environment..."
if [ ! -d "ml/.venv" ]; then
    python3 -m venv ml/.venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
source ml/.venv/bin/activate
pip install --upgrade pip --quiet
echo "✓ Virtual environment activated"
echo ""

# Step 3: Install dependencies
echo "Step 3: Installing Python dependencies (this may take 5-10 minutes)..."
if ! python -c "import tensorflow" 2>/dev/null; then
    echo "Installing TensorFlow and dependencies..."
    pip install -r ml/requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi
echo ""

# Step 4: Set up GPU environment variables
echo "Step 4: Configuring GPU environment..."
export PYTHONPATH=/workspace/optcg-builder:$PYTHONPATH

# Set CUDA library paths (try multiple common locations)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
if [ -d "/usr/local/cuda-12.8" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
elif [ -d "/usr/local/cuda-12.0" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-12.0/targets/x86_64-linux/lib:/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
elif [ -d "/usr/local/cuda-11.8" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
fi

# Add TensorFlow's bundled NVIDIA libraries
VENV_LIB=/workspace/optcg-builder/ml/.venv/lib/python3.12/site-packages
if [ -d "$VENV_LIB/nvidia" ]; then
export LD_LIBRARY_PATH=$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$VENV_LIB/nvidia/cufft/lib:$VENV_LIB/nvidia/curand/lib:$VENV_LIB/nvidia/cusolver/lib:$VENV_LIB/nvidia/cusparse/lib:$VENV_LIB/nvidia/cuda_runtime/lib:$VENV_LIB/nvidia/nccl/lib:$LD_LIBRARY_PATH
fi

echo "✓ GPU environment configured"
echo ""

# Step 5: Verify GPU detection with TensorFlow
echo "Step 5: Verifying TensorFlow GPU detection..."
python << 'PYTHON_EOF'
import sys
import os
sys.path.insert(0, '/workspace/optcg-builder')

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA built: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detected: {len(gpus)}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print("✓ GPU is accessible to TensorFlow")
    else:
        print("⚠️  No GPUs detected by TensorFlow")
        print("   This may be normal if GPU drivers aren't installed")
        print("   Training will use CPU (slower but will work)")
except Exception as e:
    print(f"⚠️  Error checking GPU: {e}")
    print("   Training will fall back to CPU")
PYTHON_EOF
echo ""

# Step 6: Check if dataset exists
echo "Step 6: Checking dataset..."
if [ -f "data/training/decks_merged.jsonl" ]; then
    echo "✓ Dataset found: data/training/decks_merged.jsonl"
elif [ -f "ml/artifacts/combined_prompt_deck.jsonl" ]; then
    echo "✓ Dataset found: ml/artifacts/combined_prompt_deck.jsonl"
    echo "  (Will use this dataset for training)"
else
    echo "⚠️  Dataset not found - you may need to create it:"
    echo "   python scripts/merge_tournament_decks.py \\"
    echo "     --tournament-root data/tournaments \\"
    echo "     --synthetic ml/artifacts/synthetic_prompt_deck.jsonl \\"
    echo "     --output data/training/decks_merged.jsonl"
fi
echo ""

# Step 7: Display ready message
echo "=== Setup Complete! Ready for Training ==="
echo ""
echo "To start training, run:"
echo ""
echo "cd /workspace/optcg-builder"
echo "source ml/.venv/bin/activate"
echo "export PYTHONPATH=/workspace/optcg-builder:\$PYTHONPATH"
echo ""
echo "# Set GPU environment (if needed)"
echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
echo "VENV_LIB=/workspace/optcg-builder/ml/.venv/lib/python3.12/site-packages"
echo "export LD_LIBRARY_PATH=\$VENV_LIB/nvidia/cudnn/lib:\$VENV_LIB/nvidia/cublas/lib:\$VENV_LIB/nvidia/cufft/lib:\$VENV_LIB/nvidia/curand/lib:\$VENV_LIB/nvidia/cusolver/lib:\$VENV_LIB/nvidia/cusparse/lib:\$VENV_LIB/nvidia/cuda_runtime/lib:\$VENV_LIB/nvidia/nccl/lib:\$LD_LIBRARY_PATH"
echo ""
echo "# Or simply run:"
echo "bash train_model.sh"
echo ""
echo "Training command:"
echo "python -m ml.training.train \\"
echo "  --dataset data/training/decks_merged.jsonl \\"
echo "  --output-dir models \\"
echo "  --epochs 40 \\"
echo "  --batch-size 32 \\"
echo "  --learning-rate 3e-4 \\"
echo "  --freq-hist-weight 20.0 \\"
echo "  --entropy-penalty 3.0 \\"
echo "  --low-prob-penalty 15.0 \\"
echo "  --low-prob-threshold 0.25 \\"
echo "  --coverage-weight 0.05 \\"
echo "  --disable-checkpoints"
echo ""

