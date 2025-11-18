#!/bin/bash
# Training script for RunPod RTX 4090
# Make sure you've run the setup script first

cd /workspace/optcg-builder
source ml/.venv/bin/activate

# Set environment variables for GPU
export PYTHONPATH=/workspace/optcg-builder:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
VENV_LIB=/workspace/optcg-builder/ml/.venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$VENV_LIB/nvidia/cufft/lib:$VENV_LIB/nvidia/curand/lib:$VENV_LIB/nvidia/cusolver/lib:$VENV_LIB/nvidia/cusparse/lib:$VENV_LIB/nvidia/cuda_runtime/lib:$VENV_LIB/nvidia/nccl/lib:$LD_LIBRARY_PATH

# Run training with Phase 2: Autoregressive sequence-level loss
# Phase 2 generates sequences autoregressively during training (not just argmax from teacher forcing)
# This directly penalizes singleton-heavy generated sequences
# Phase 2 is enabled by default (use --disable-phase2 to disable)
python -m ml.training.train \
  --dataset data/training/decks_merged.jsonl \
  --output-dir models \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --freq-hist-weight 100.0 \
  --entropy-penalty 2.0 \
  --low-prob-penalty 10.0 \
  --low-prob-threshold 0.3 \
  --sequence-level-weight 200.0 \
  --phase2-scheduled-sampling-rate 0.5 \
  --phase2-generation-batch-fraction 0.25 \
  --coverage-weight 0.05 \
  --early-stopping-patience 10 \
  --disable-checkpoints
