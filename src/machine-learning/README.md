# Task 1 — Facial Engagement Detection

Real-time engagement detection using DAiSEE + EfficientNet-B0 + Temporal Transformer.

---

## File Map

```
task1/
├── config.py               Single source of truth for all paths & hyperparameters
├── preprocess_daisee.py    Extract face crops from DAiSEE videos → .npy files
├── dataset.py              PyTorch Dataset + DataLoader factory
├── model.py                CNN backbone + Temporal Transformer architecture
├── train.py                Training loop (AMP, warmup, checkpointing, TensorBoard)
├── export_onnx.py          Export checkpoint → ONNX → INT8 quantised + benchmark
├── realtime_inference.py   Webcam loop using ONNX Runtime (runs on demo laptop)
├── requirements.txt        Demo laptop dependencies (inference only)
└── slurm/
    ├── preprocess.slurm    Slurm job: extract face crops (CPU, ~90 min)
    └── train.slurm         Slurm job: train model on H100 (~90 min for 40 epochs)
```

---

## Step-by-Step

### Step 0 — Before you touch any code

1. Get DAiSEE access at https://iith.ac.in/~daisee-dataset/ (requires academic email)
2. Confirm your Gautschi allocation is active: `ssh gautschi.rcac.purdue.edu`
3. Upload DAiSEE to scratch: `rsync -av DAiSEE/ gautschi:/scratch/$USER/DAiSEE/` / put DAiSEE in proper location
---

### Step 1 — Set up environment on Gautschi

```bash 
ssh gautschi.rcac.purdue.edu
cd /scratch/$USER
git clone <your_repo> clippy && cd clippy/

# Edit config.py: set GAUTSCHI_USER = "your_actual_username"
nano config.py

bash setup_env.sh
```

---

### Step 2 — Preprocess DAiSEE
In preprocess.slurm fix the email

```bash
mkdir -p logs
sbatch slurm/preprocess.slurm

# Monitor progress
tail -f logs/preprocess_$(squeue -u $USER -h -o %i).out
```

Expected output in `CROPS_ROOT/{train,val,test}/`:
- `{clip_id}/frames.npy`  — shape `(N, 112, 112, 3)` uint8
- `{clip_id}/labels.npy`  — shape `(4,)` float32 in `[0, 1]`
- `manifest.json`          — clip list with frame counts

Typical durations:
- train split: ~60 min with 32 workers
- val + test:  ~15 min each

---

### Step 3 — Verify preprocessing

```bash
conda activate clippy
uv run dataset.py
# Expected output:
# Frames shape : torch.Size([10, 3, 112, 112])
# Labels shape : torch.Size([4])
# Label values : tensor([0.33, 0.67, 0.00, 0.00])
# Dataset OK
```

---

### Step 4 — Train

In train.slurm fix the email
```bash
sbatch slurm/train.slurm
tail -f logs/train_<JOBID>.out
```

Expected training time: **~90 minutes on one H100** for 40 epochs.

If val_loss is not improving after epoch 20, check that preprocessing
completed without errors and that labels are loaded correctly.

---

### Step 5 — Export to ONNX

The `train.slurm` script runs this automatically at the end. To run manually:

```bash
conda activate clippy
python export_onnx.py \
    --ckpt /scratch/gautschi/$USER/clippy/clippy_checkpoints/best.pt \
    --quantize \
    --benchmark
```

Look for this in the output:
```
Mean latency : 35.2 ms
[OK] Under 80 ms target — real-time capable on demo laptop
```

---

### Step 6 — Copy to demo laptop

```bash
# From your laptop terminal:
scp user@gautschi.rcac.purdue.edu:/scratch/gautschi/$USER/folder/clippy_engagement_int8.onnx ~/clippy/
scp user@gautschi.rcac.purdue.edu:/scratch/gautschi/$USER/folder/config.py ~/clippy/
scp user@gautschi.rcac.purdue.edu:/scratch/gautschi/$USER/folder/realtime_inference.py ~/clippy/

# Install inference dependencies on laptop
uv sync
```

---

### Step 7 — Test real-time inference on demo laptop

```bash
# In ~/clippy/ directory with the .onnx file and realtime_inference.py
python realtime_inference.py
```

A webcam window opens with live emotion bars. Terminal prints:
```
[14:32:01] eng=0.71  bor=0.12  con=0.22  fru=0.09  face=True  → engaged
```

Press `q` in the video window to quit.

---

## Troubleshooting

**"ONNX model not found"**
→ Copy `clippy_engagement_int8.onnx` from Gautschi and set `ONNX_INT8` in `config.py`.

**"Manifest not found"**
→ Preprocessing hasn't run yet. Check `squeue` and preprocess logs.

**Face not detected**
→ Improve lighting. MTCNN `min_face_size=40` requires the face to occupy at
  least 40×40 pixels — move closer to the webcam or lower this threshold in `config.py`.

**Inference > 80 ms on laptop**
→ Ensure you're using `clippy_engagement_int8.onnx`, not the FP32 version.
  If still slow, reduce `seq_len` from 10 to 6 in `config.py` (retrain required).

**Training loss not decreasing**
→ Verify label range: `np.load("labels.npy")` should be in `[0, 1]`, not `[0, 3]`.
  Check that `LABEL_COLS` in `config.py` matches the actual CSV column names in your DAiSEE version.

---

## Output contract (for Task 3 and Task 4)

`realtime_inference.py` emits `EngagementState` objects every ~3 seconds onto a `queue.Queue`:

```python
@dataclass
class EngagementState:
    timestamp:    float   # Unix time
    boredom:      float   # [0, 1]
    engagement:   float   # [0, 1]
    confusion:    float   # [0, 1]
    frustration:  float   # [0, 1]
    confidence:   float   # MTCNN detection confidence [0, 1]
    face_detected: bool
```

Task 3 reads from the queue to select music.
Task 4 reads from the queue to track long-term fatigue.
