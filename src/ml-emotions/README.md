# 🎭 ml-emotions-alt — YOLOv8 Emotion Classifier

Real-time facial emotion detection using **YOLOv8x-cls** trained on AffectNet.
Runs on Purdue **Gautschi** (8× NVIDIA H100 SXM, 80 GB each, NVLinked).

**8 emotions:** anger · contempt · disgust · fear · happy · neutral · sad · surprise

---

## 📁 Project Structure

```
ml-emotions-alt/
├── data/
│   ├── train/          ← AffectNet training images (class subfolders)
│   └── val/            ← AffectNet validation images (class subfolders)
├── slurm/
│   ├── train.sh        ← Submit training job (sbatch)
│   ├── validate.sh     ← Submit validation job
│   └── debug.sh        ← Launch interactive GPU session
├── outputs/            ← Training runs + validation reports saved here
├── logs/               ← SLURM stdout/stderr logs
├── prepare_data.py     ← Fix val/ directory case issues (run first!)
├── train.py            ← Training script
├── validate.py         ← Full evaluation with plots
├── webcam_test.py      ← Real-time webcam demo (run LOCALLY)
└── requirements.txt
```

---

## ⚙️ Setup (one-time)

### 1. Find your Gautschi account name
```bash
sacctmgr show associations user=$USER format=account
```
Update `--account=catapult` in all `slurm/*.sh` files with your account name.

### 2. Check/confirm your conda environment
```bash
conda activate clippy
conda env list           # verify 'clippy' exists
module avail learning    # see available ML modules
```
Update `module load learning/...` in slurm scripts if the module name differs.

### 3. Install Python dependencies
```bash
conda activate clippy
pip install -r requirements.txt
```

### 4. Verify data location
```bash
ls data/train    # should show: anger contempt disgust fear happy neutral sad surprise
ls data/val      # may show mixed case — prepare_data.py will fix this
```

---

## 🚀 Training

### Step 1 — Fix data (run once)
```bash
# Preview changes without modifying files
python prepare_data.py --data_dir ./data --dry_run

# Apply fixes
python prepare_data.py --data_dir ./data
```

### Step 2 — Submit training job
```bash
sbatch slurm/train.sh

# Monitor
squeue -u $USER
tail -f logs/train_<JOBID>.out
```

**What this does:**
- Runs on 1 Gautschi-H node with all **8 H100 GPUs** (NVLinked)
- Auto-scales batch size to fill GPU memory (~512/GPU → 4096 effective)
- Caches dataset in RAM (1 TB available) for max throughput
- Mixed precision (BF16) — H100 native
- Strong augmentation: RandAugment + random erasing
- Early stopping with patience=20
- Checkpoints every 10 epochs → `outputs/emotion_yolov8x/weights/`

### Resume from checkpoint
```bash
# Edit train.py call in slurm/train.sh, add:
#   --resume outputs/emotion_yolov8x/weights/last.pt
```

---

## 📊 Validation

```bash
sbatch slurm/validate.sh
```

Outputs in `outputs/validation/`:
- `report_val.txt` — accuracy, precision, recall, F1 per class
- `metrics_val.json` — machine-readable metrics
- `confusion_matrix_val.png` — raw + normalized confusion matrix
- `per_class_accuracy_val.png` — bar chart by emotion
- `misclassified/` — error gallery (top-20 wrong predictions per class)

**Quick check on login node (small sample):**
```bash
# Don't run heavy inference on login node — use interactive session
bash slurm/debug.sh
# Inside interactive session:
python validate.py --model outputs/emotion_yolov8x/weights/best.pt --batch 64
```

---

## 🎥 Webcam Demo (run locally)

### 1. Copy model weights to your laptop
```bash
# On your LOCAL machine:
scp pham191@gautschi.rcac.purdue.edu:/scratch/gautschi/pham191/\
Catapult-2026-Best-Ever-Project/src/ml-emotions-alt/\
outputs/emotion_yolov8x/weights/best.pt ./best.pt
```

### 2. Install local dependencies
```bash
pip install ultralytics opencv-python pillow numpy
```

### 3. Run
```bash
python webcam_test.py --model best.pt          # default webcam
python webcam_test.py --model best.pt --camera 1    # alternate camera
python webcam_test.py --model best.pt --video clip.mp4  # from file
python webcam_test.py --model best.pt --device 0   # use local GPU if available
```

**Controls:**
- `q` — quit
- `s` — save screenshot
- `f` — toggle face detection confidence display

---

## 🔧 Tuning Tips

| Goal | Change |
|---|---|
| Higher accuracy | Increase `--imgsz 320`, add more epochs |
| Faster training | Reduce model to `yolov8l-cls.pt` |
| Reduce overfitting | Increase `--dropout 0.3`, `--label_smoothing 0.15` |
| Faster webcam | Use `yolov8n-cls.pt` for a lightweight export |

### Export to ONNX (faster CPU inference)
```bash
yolo export model=outputs/emotion_yolov8x/weights/best.pt format=onnx imgsz=224
```

---

## 📈 Expected Results

On AffectNet, YOLOv8x-cls typically achieves:
- **Overall Top-1 accuracy:** ~72–78% (8-class)
- **Happy/Neutral:** >85% (abundant data, distinct features)
- **Contempt/Fear:** ~55–65% (harder; limited data, subtle expressions)

The confusion matrix will reveal which emotions are most commonly confused.

---

## 🐛 Troubleshooting

**"CUDA out of memory"** — Edit `train.py` auto_batch or add `--batch 256`

**"No module named ultralytics"** — `pip install ultralytics` in your conda env

**val/ folder issues** — Re-run `python prepare_data.py --data_dir ./data`

**SLURM account error** — Run `sacctmgr show assoc user=$USER` and update account name

**Webcam not opening** — Try `--camera 1` or check `ls /dev/video*`
