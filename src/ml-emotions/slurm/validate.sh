#!/bin/bash
# ============================================================
# slurm/validate.sh
# SLURM job: validate trained emotion classifier
#
# Submit: sbatch slurm/validate.sh
# ============================================================

#SBATCH --job-name=emotion-validate
#SBATCH --partition=ai
#SBATCH --account=mlp          # ← replace with your account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1                # 1 H100 is plenty for inference
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pham191@purdue.edu

set -euo pipefail

WORKDIR="/scratch/gautschi/pham191/Catapult-2026-Best-Ever-Project/src/ml-emotions-alt"
cd "$WORKDIR"

echo "Job ID: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  $(date)"

module purge
module load learning/conda-2024.06-py311-gpu
conda activate clippy

# ── Validate on val split ────────────────────────────────────────────────────
echo ""
echo "=== Validating on: val ==="
python validate.py \
    --model    outputs/emotion_yolov8x/weights/best.pt \
    --data_dir ./data \
    --split    val \
    --imgsz    224 \
    --batch    256 \
    --device   0 \
    --out_dir  outputs/validation \
    --top_k    20

# ── Validate on train split (to check for over/underfitting) ─────────────────
echo ""
echo "=== Validating on: train ==="
python validate.py \
    --model    outputs/emotion_yolov8x/weights/best.pt \
    --data_dir ./data \
    --split    train \
    --imgsz    224 \
    --batch    256 \
    --device   0 \
    --out_dir  outputs/validation \
    --top_k    10

echo ""
echo "=== Validation complete at $(date) ==="
echo "Results: $WORKDIR/outputs/validation/"
