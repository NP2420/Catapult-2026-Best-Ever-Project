#!/bin/bash
# ============================================================
# slurm/train.slurm
#
# Trains EngagementModel on DAiSEE using one H100 GPU.
# Preprocessing (preprocess.slurm) must complete first.
#
# Submit : sbatch slurm/train.slurm
# Monitor: squeue -u $USER
# Watch  : tail -f logs/train_<JOBID>.out
#
# To run multiple experiments in parallel, submit this script
# several times with different --run_name arguments.
# ============================================================

#SBATCH --job-name=clippy_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # DataLoader workers
#SBATCH --gres=gpu:1              # 1 × H100-80GB is plenty for this model
#SBATCH --mem=64G                 # CPU RAM for DataLoader prefetching
#SBATCH --time=04:00:00           # 40 epochs ≈ 90 min on H100; 4 h is safe headroom
#SBATCH --partition=ai            # Gautschi-AI partition (H100 nodes)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@purdue.edu   # <-- change this

mkdir -p logs

echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU(s)    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA      : $(nvcc --version | tail -1)"
echo "Start     : $(date)"
echo "============================================"

module load anaconda
source activate clippy

echo "Python    : $(which python)"
echo "PyTorch   : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name  : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

# ---- Experiment 1: baseline (submit this first) ----
python train.py \
    --epochs      40 \
    --batch_size  256 \
    --lr          3e-4 \
    --num_workers 16 \
    --run_name    "effb0_t10_d256"

# ---- After baseline finishes, submit a second job with this block ----
# (comment out the above and uncomment below)
#
# python train.py \
#     --epochs      40 \
#     --batch_size  256 \
#     --lr          3e-4 \
#     --num_workers 16 \
#     --run_name    "effb2_t15_d256"

echo "============================================"
echo "End       : $(date)"
echo "============================================"

# Auto-export best checkpoint to ONNX after training
echo "Exporting best checkpoint to ONNX..."
python export_onnx.py \
    --ckpt    /scratch/$USER/clippy_checkpoints/best.pt \
    --quantize \
    --benchmark

echo "Copy to demo laptop with:"
echo "  scp gautschi.rcac.purdue.edu:/scratch/$USER/clippy_engagement_int8.onnx ~/clippy/"
