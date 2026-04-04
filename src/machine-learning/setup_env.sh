#!/bin/bash
# ============================================================
# setup_env.sh — Run this ONCE on a Gautschi login node
# Usage: bash setup_env.sh
# ============================================================

set -e  # exit on any error

ENV_NAME="clippy"
PYTHON_VERSION="3.11"

echo "==> Loading Anaconda module..."
module load anaconda

echo "==> Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "==> Activating environment..."
source activate $ENV_NAME

echo "==> Installing PyTorch with CUDA 12.1 (matches Gautschi H100 drivers)..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y

echo "==> Installing training dependencies..."
pip install \
    timm==0.9.16 \
    facenet-pytorch==2.6.0 \
    albumentations==1.4.3 \
    opencv-python-headless==4.9.0.80 \
    einops==0.7.0 \
    pandas==2.2.1 \
    scikit-learn==1.4.1 \
    tqdm==4.66.2 \
    tensorboard==2.16.2 \
    matplotlib==3.8.3 \
    seaborn==0.13.2

echo "==> Installing ONNX export + inference dependencies..."
pip install \
    onnx==1.16.0 \
    onnxruntime==1.17.1 \
    onnxruntime-tools==1.7.0

echo "==> Installing real-time inference dependencies..."
pip install \
    plyer==2.1.0

echo ""
echo "==> Setup complete. Activate with: conda activate $ENV_NAME"
echo "==> Test with: python -c \"import torch; print(torch.cuda.is_available())\""
