#!/bin/bash
# Script to zip and prepare model for download from RunPod
# Usage: bash download_model.sh [model_path]

MODEL_PATH="${1:-models/run_20251117-054447}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Usage: bash download_model.sh [model_path]"
    exit 1
fi

echo "=== Preparing Model for Download ==="
echo "Model path: $MODEL_PATH"
echo ""

# Get the run directory name
RUN_NAME=$(basename "$MODEL_PATH")
PARENT_DIR=$(dirname "$MODEL_PATH")

# Create zip file name with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ZIP_NAME="${RUN_NAME}_${TIMESTAMP}.zip"

echo "Creating zip file: $ZIP_NAME"
echo ""

# Zip the model directory (excluding large cache files if any)
cd "$PARENT_DIR"
zip -r "$ZIP_NAME" "$RUN_NAME" \
    -x "*.pyc" \
    -x "__pycache__/*" \
    -x "*.log" \
    -x "checkpoints/*.index" \
    -x "checkpoints/*.data-*" \
    -x "*.tmp"

if [ $? -eq 0 ]; then
    ZIP_SIZE=$(du -h "$ZIP_NAME" | cut -f1)
    echo ""
    echo "✓ Zip file created successfully!"
    echo "  File: $PARENT_DIR/$ZIP_NAME"
    echo "  Size: $ZIP_SIZE"
    echo ""
    echo "=== Download Instructions ==="
    echo ""
    echo "Option 1: Download via RunPod Web UI"
    echo "  1. Go to RunPod dashboard"
    echo "  2. Open your pod's file browser"
    echo "  3. Navigate to: $PARENT_DIR"
    echo "  4. Download: $ZIP_NAME"
    echo ""
    echo "Option 2: Download via terminal (if scp/sftp is configured)"
    echo "  scp root@<runpod-ip>:/workspace/optcg-builder/$PARENT_DIR/$ZIP_NAME ."
    echo ""
    echo "Option 3: Use RunPod's download feature"
    echo "  The file is located at: /workspace/optcg-builder/$PARENT_DIR/$ZIP_NAME"
    echo ""
else
    echo "Error: Failed to create zip file"
    exit 1
fi

