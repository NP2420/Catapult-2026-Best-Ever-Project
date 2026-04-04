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
#SBATCH --mail-user=your_email@purdue.edu   # <-- change this

# Create log directory if it doesn't exist
mkdir -p logs

echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Start     : $(date)"
echo "============================================"

# Load conda and activate environment
module load conda
conda activate mypackages

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
