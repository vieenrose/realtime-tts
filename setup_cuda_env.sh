#!/bin/bash
# Set up CUDA/cuDNN library paths for ONNX Runtime GPU

VENV_DIR="${VENV_DIR:-$(dirname $0)/.venv}"

export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV_DIR/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV_DIR/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

echo "CUDA library paths configured:"
echo "  cuDNN: $VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib"
echo "  cuBLAS: $VENV_DIR/lib/python3.12/site-packages/nvidia/cublas/lib"
echo "  NVRTC: $VENV_DIR/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"