"""
config.py — Single source of truth for all paths and hyperparameters.

Edit the PATHS section to match your Gautschi scratch directory before running
any other script.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# PATHS — edit these before running anything
# ============================================================

GAUTSCHI_USER = "pham191"  # <-- change this

DAISEE_ROOT   = Path(f"/scratch/{GAUTSCHI_USER}/DAiSEE")
CROPS_ROOT    = Path(f"/scratch/{GAUTSCHI_USER}/DAiSEE_crops")
CKPT_DIR      = Path(f"/scratch/{GAUTSCHI_USER}/clippy_checkpoints")
LOGS_DIR      = Path(f"/scratch/{GAUTSCHI_USER}/clippy_logs")
ONNX_PATH     = Path(f".\onxx\clippy_engagement.onnx")
ONNX_INT8     = Path(f".\onxx\clippy_engagement_int8.onnx")

# DAiSEE internal structure (do not change unless your copy differs)
DAISEE_LABELS = {
    "train": DAISEE_ROOT / "Labels" / "TrainLabels.csv",
    "val":   DAISEE_ROOT / "Labels" / "ValidationLabels.csv",
    "test":  DAISEE_ROOT / "Labels" / "TestLabels.csv",
}
DAISEE_VIDEOS = {
    "train": DAISEE_ROOT / "DataSet" / "Train",
    "val":   DAISEE_ROOT / "DataSet" / "Validation",
    "test":  DAISEE_ROOT / "DataSet" / "Test",
}

# Label columns in DAiSEE CSVs (index 0–3 maps to our model's 4 outputs)
LABEL_COLS  = ["Boredom", "Engagement", "Confusion", "Frustration"]
LABEL_NAMES = ["boredom", "engagement", "confusion", "frustration"]
NUM_LABELS  = 4


# ============================================================
# PREPROCESSING
# ============================================================

@dataclass
class PreprocConfig:
    target_fps: int   = 5       # extract every Nth frame to reach this rate
    crop_size:  int   = 112     # face crop resolution (HxW); 112 fits EfficientNet-B0
    margin:     int   = 20      # MTCNN face crop margin in pixels
    min_frames: int   = 5       # discard clips with fewer detected faces than this
    num_workers: int  = 16      # parallel video workers in the Slurm CPU job


# ============================================================
# MODEL
# ============================================================

@dataclass
class ModelConfig:
    backbone:   str = "efficientnet_b0"  # timm model name; swap to b2 for +~1% acc
    pretrained: bool = True              # use ImageNet weights as starting point
    d_model:    int  = 256              # Transformer hidden dimension
    n_heads:    int  = 4                # attention heads (d_model must be divisible)
    n_layers:   int  = 2                # Transformer encoder depth
    seq_len:    int  = 10               # frames per sequence (2 s at 5 fps)
    dropout:    float = 0.1
    num_outputs: int = NUM_LABELS


# ============================================================
# TRAINING
# ============================================================

@dataclass
class TrainConfig:
    epochs:        int   = 40
    batch_size:    int   = 256    # fits one H100-80GB easily; raise to 512 if memory allows
    lr:            float = 3e-4
    weight_decay:  float = 1e-4
    warmup_epochs: int   = 3      # linear LR warmup before cosine decay
    grad_clip:     float = 1.0    # gradient clipping max norm
    label_smooth:  float = 0.05   # soft targets to reduce overconfidence
    num_workers:   int   = 16
    pin_memory:    bool  = True
    save_every:    int   = 5      # checkpoint every N epochs
    mixed_precision: bool = True  # use torch.cuda.amp (free ~30% speedup on H100)

    # Class-imbalance correction: DAiSEE has far fewer high-engagement clips
    # We oversample them during DataLoader construction
    oversample_engaged: bool = True
    engaged_threshold:  float = 0.67   # clips with engagement > this get extra weight


# ============================================================
# REAL-TIME INFERENCE
# ============================================================

@dataclass
class InferenceConfig:
    onnx_path:      Path  = ONNX_INT8   # use quantized model by default
    emit_interval:  float = 3.0         # seconds between emitting an EngagementState
    ema_alpha:      float = 0.2         # smoothing factor (lower = smoother)
    min_confidence: float = 0.50        # skip emit if face detection confidence < this
    cam_index:      int   = 0           # webcam device index
    display_hud:    bool  = True        # overlay emotion bars on webcam feed

    # ImageNet normalisation applied to face crops
    mean: tuple = (0.485, 0.456, 0.406)
    std:  tuple = (0.229, 0.224, 0.225)


# ============================================================
# CONVENIENCE: single config object
# ============================================================

@dataclass
class Config:
    preproc:   PreprocConfig   = field(default_factory=PreprocConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    train:     TrainConfig     = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


CFG = Config()
