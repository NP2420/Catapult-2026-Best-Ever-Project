#!/bin/bash
# ============================================================
# slurm/train.sh
# SLURM job: train emotion classifier on all 8x H100 GPUs
# Partition: ai (Gautschi-AI, H100 SXM 80 GB × 8)
#
# Submit: sbatch slurm/train.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/train_<JOBID>.out
# ============================================================

#SBATCH --job-name=emotion-train
#SBATCH --partition=ai
#SBATCH --account=mlp          # ← replace with your account (sacctmgr show assoc user=$USER)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112         # all cores on a Gautschi-H node (112 / 8 GPUs = 14/GPU)
#SBATCH --gres=gpu:8                # all 8 H100s (NVLinked)
#SBATCH --mem=800G                  # ~9 GB/core × 112 cores ≈ 1 TB available; use 800 to be safe
#SBATCH --time=24:00:00             # 24 h (adjust per your GPU-hour budget)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pham191@purdue.edu

# ── Environment ──────────────────────────────────────────────────────────────
set -euo pipefail

WORKDIR="/scratch/gautschi/pham191/Catapult-2026-Best-Ever-Project/src/ml-emotions-alt"
cd "$WORKDIR"

echo "========================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Work dir   : $WORKDIR"
echo "Start time : $(date)"
echo "========================================================"

# ── Load modules & activate environment ──────────────────────────────────────
module purge
module load learning/conda-2024.06-py311-gpu  # adjust to available module; check: module avail learning

# Activate your conda environment (check: conda env list)
conda activate clippy

# Verify GPU access
echo ""
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Check ultralytics is installed ───────────────────────────────────────────
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

# ── Step 1: Prepare data (idempotent — safe to re-run) ───────────────────────
echo ""
echo "=== Step 1: Data Preparation ==="
python prepare_data.py --data_dir ./data

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Training ==="
python train.py \
    --data_dir  ./data \
    --model     yolov8x-cls.pt \
    --epochs    100 \
    --imgsz     224 \
    --batch     -1 \
    --workers   14 \
    --gpus      0,1,2,3,4,5,6,7 \
    --lr        0.001 \
    --patience  20 \
    --name      emotion_yolov8x \
    --label_smoothing 0.1 \
    --dropout   0.2

echo ""
echo "=== Training complete at $(date) ==="
echo "Best weights: $WORKDIR/outputs/emotion_yolov8x/weights/best.pt"
