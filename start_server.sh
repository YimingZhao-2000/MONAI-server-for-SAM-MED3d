#!/bin/bash

# SAM-Med3D MONAI Label Server Startup Script

echo "Starting SAM-Med3D MONAI Label Server..."

# Set default values
MODEL_TYPE=${SAM_MODEL_TYPE:-"vit_b_ori"}
CHECKPOINT_PATH=${SAM_CHECKPOINT_PATH:-"ckpt/sam_med3d.pth"}
DATA_PATH=${MONAI_LABEL_DATASTORE:-"./data"}
PORT=${MONAI_LABEL_PORT:-"8000"}

# Check if model exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Warning: Model file not found at $CHECKPOINT_PATH"
    echo "Please set SAM_CHECKPOINT_PATH environment variable to point to your model file"
fi

# Start MONAI Label server
echo "Starting server on port $PORT..."
echo "Data path: $DATA_PATH"
echo "Model type: $MODEL_TYPE"
echo "Checkpoint path: $CHECKPOINT_PATH"

monailabel start_server \
  --app . \
  --studies /path/to/data \
  --conf models deepedit