#!/bin/bash
# ============================================================
# slurm/train.slurm
# Submit : sbatch slurm/train.slurm
# Watch  : tail -f logs/train_<JOBID>.out
# ============================================================

#SBATCH --job-name=clippy_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14        # Gautschi AI: exactly 14 CPUs per GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00           # 60 epochs ≈ 110 min on H100; 6h is safe
#SBATCH --partition=ai
#SBATCH --account=mlp
#SBATCH --qos=normal
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pham191@purdue.edu

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

python train.py \
    --epochs      60 \
    --batch_size  32 \
    --lr          3e-4 \
    --num_workers 14 \
    --run_name    "run_02_weighted_e60"

echo "============================================"
echo "End       : $(date)"
echo "============================================"

echo "Exporting best checkpoint to ONNX..."
python export_onnx.py \
    --ckpt    /scratch/gautschi/pham191/clippy/clippy_checkpoints/best.pt \
    --quantize \
    --benchmark

echo "Copy to demo laptop with:"
echo "  scp gautschi.rcac.purdue.edu:/scratch/gautschi/pham191/clippy/clippy_engagement_int8.onnx ~/clippy/"

(clippy) pham191@login00.gautschi:[clippy] $ cat slurm/preprocess.slurm
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
#SBATCH --mail-user=pham191@purdue.edu   # <-- change this

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