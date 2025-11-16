#!/bin/bash
# Training script for RunPod RTX 4090
# Make sure you've run the setup script first

cd /workspace/optcg-builder
source ml/.venv/bin/activate

# Set environment variables
export PYTHONPATH=/workspace/optcg-builder:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
VENV_LIB=/workspace/optcg-builder/ml/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$VENV_LIB/nvidia/cufft/lib:$VENV_LIB/nvidia/curand/lib:$VENV_LIB/nvidia/cusolver/lib:$VENV_LIB/nvidia/cusparse/lib:$VENV_LIB/nvidia/cuda_runtime/lib:$VENV_LIB/nvidia/nccl/lib:$LD_LIBRARY_PATH

# Run training
python -m ml.training.train \
  --dataset ml/artifacts/combined_prompt_deck.jsonl \
  --output-dir models \
  --epochs 40 \
  --batch-size 48 \
  --learning-rate 2e-4 \
  --coverage-weight 0.05 \
  --freq-hist-weight 15.0 \
  --low-prob-penalty 8.0

