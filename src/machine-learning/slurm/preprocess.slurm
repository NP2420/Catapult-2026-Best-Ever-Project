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
#SBATCH --cpus-per-task=14        # DataLoader workers
#SBATCH --gres=gpu:1              # 1 × H100-80GB is plenty for this model
#SBATCH --mem=64G                 # CPU RAM for DataLoader prefetching
#SBATCH --time=04:00:00           # 40 epochs ≈ 90 min on H100; 4 h is safe headroom
#SBATCH --partition=ai            # Gautschi-AI partition (H100 nodes)
#SBATCH --account=mlp
#SBATCH --qos=normal
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pham191@purdue.edu   # <-- change this

mkdir -p logs

echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU(s)    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA      : $(python -c 'import torch; print(torch.version.cuda)')"
echo "Start     : $(date)"
echo "============================================"

module load conda
conda activate clippy

echo "Python    : $(which python)"
echo "PyTorch   : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name  : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

# ---- Experiment 1: baseline (submit this first) ----
python train.py \
    --epochs      40 \
    --batch_size  256 \
    --lr          3e-4 \
    --num_workers 14 \
    --run_name    "effb0_t10_d256"

# ---- After baseline finishes, submit a second job with this block ----
# (comment out the above and uncomment below)
#
# python train.py \
#     --epochs      40 \
#     --batch_size  256 \
#     --lr          3e-4 \
#     --num_workers 14 \
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
pham191@login00.gautschi:[clippy] $ cat slurm/preprocess.slurm
#!/bin/bash
# ============================================================
# slurm/preprocess.slurm
#
# Extracts face crops from all DAiSEE video clips.
# Run ONCE before training. No GPU needed.
#
# Submit: sbatch slurm/preprocess.slurm
# Monitor: squeue -u $USER
# Logs:    cat logs/preprocess_<JOBID>.out
# ============================================================

#SBATCH --job-name=daisee_preproc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32        # 32 parallel video workers
#SBATCH --mem=64G                 # ~7MB per clip × 9068 clips ≈ 63 GB peak
#SBATCH --time=03:00:00           # 3 hours is generous; usually finishes in ~90 min
#SBATCH --partition=cpu           # CPU partition — no GPU needed
#SBATCH --account=mlp
#SBATCH --qos=normal
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pham191@purdue.edu #HEREEEE

# Create log directory if it doesn't exist
mkdir -p logs

echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Start     : $(date)"
echo "============================================"

# Load conda and activate environment
module load conda
conda activate clippy

# Verify environment
echo "Python    : $(which python)"
echo "PyTorch   : $(python -c 'import torch; print(torch.__version__)')"

# Run preprocessing for all splits
python preprocess_daisee.py \
    --splits train val test \
    --workers 32

echo "============================================"
echo "End       : $(date)"
echo "============================================"