# MONAI-server-for-SAM-MED3d

This project provides a MONAI Label server integration for [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D), enabling interactive 3D medical image segmentation with clients such as 3D Slicer.

## Main Contents
- `config.py`: MONAI Label App configuration, integrating SAM-Med3D inference logic
- `infer_utils.py`: Inference and post-processing utilities, supporting user click input
- `requirements.txt`: Dependency list
- `start_server.sh`: One-click script to start the MONAI Label server

## Quick Start

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare model weights
   ```bash
   export SAM_CHECKPOINT_PATH="ckpt/sam_med3d_turbo.pth"
   ```
3. Start the server
   ```bash
   ./start_server.sh
   ```
4. Connect to this service using the MONAI Label plugin in 3D Slicer for interactive segmentation.

## Dependencies
- torch
- torchvision
- numpy
- nibabel
- monai-label
- torchio
- medim

## Notes
- Inference supports user-provided clicks (`points`/`labels`) from Slicer clients, and also falls back to center-click segmentation if no interaction is provided.
- The code follows the Occam's razor principle, maximizing reuse of the original SAM-Med3D inference logic.

---
Feel free to open issues or pull requests if you have any questions!
