"""
export_onnx.py — Export a trained checkpoint to ONNX and optionally quantise to INT8.

Run this on Gautschi after training, then copy the .onnx file to your demo laptop.

Usage:
    python export_onnx.py                          # uses config defaults
    python export_onnx.py --ckpt checkpoints/best.pt --quantize --benchmark

The script:
  1. Loads the checkpoint onto CPU
  2. Exports to ONNX (opset 17)
  3. Validates that ONNX output matches PyTorch output (tolerance check)
  4. Optionally runs dynamic INT8 quantisation (weight-only, no calibration data needed)
  5. Benchmarks ONNX Runtime latency on CPU with a realistic batch size of 1

Target: <80 ms per inference call on a modern laptop CPU (i5/i7 from 2020+).
If you exceed this, try the INT8 model or reduce seq_len in config.py.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from config import CFG, CKPT_DIR, ONNX_INT8, ONNX_PATH
from model import EngagementModel


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_onnx(model: EngagementModel, output_path: Path) -> None:
    model.eval()

    seq_len = CFG.model.seq_len
    crop_sz = CFG.preproc.crop_size
    dummy   = torch.randn(1, seq_len, 3, crop_sz, crop_sz)

    print(f"Exporting to ONNX → {output_path}")
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["frames"],
        output_names=["scores"],
        dynamic_axes={
            "frames": {0: "batch_size"},   # allow batch size 1 or N at runtime
            "scores": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,          # folds constants for faster inference
        export_params=True,
    )
    print("  Export done.")

    # Structural validation
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure: OK")

    # Numerical validation: compare ONNX output to PyTorch output
    sess    = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    pt_out  = model(dummy).detach().numpy()
    ort_out = sess.run(None, {"frames": dummy.numpy()})[0]
    max_diff = np.abs(pt_out - ort_out).max()
    print(f"  Max absolute difference (PyTorch vs ONNX): {max_diff:.6f}")

    if max_diff > 1e-3:
        print("  WARNING: diff > 1e-3 — check for unsupported ops or precision issues")
    else:
        print("  Numerical validation: PASS")


# ---------------------------------------------------------------------------
# INT8 Quantisation
# ---------------------------------------------------------------------------

def quantize_model(input_path: Path, output_path: Path) -> None:
    """
    Dynamic quantisation: quantises weights to INT8 at export time.
    No calibration dataset needed, which makes it ideal for hackathons.
    Activations are quantised dynamically at runtime.

    Expected result: ~40-60% faster inference, <1% accuracy drop.
    """
    print(f"\nQuantising to INT8 → {output_path}")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("  onnxruntime-tools not installed. Run: pip install onnxruntime-tools")
        return

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )

    # Verify the quantised model is valid
    onnx.checker.check_model(onnx.load(str(output_path)))
    size_fp32 = input_path.stat().st_size / 1e6
    size_int8 = output_path.stat().st_size / 1e6
    print(f"  FP32 model size: {size_fp32:.1f} MB")
    print(f"  INT8 model size: {size_int8:.1f} MB  ({size_int8/size_fp32*100:.0f}%)")
    print("  Quantisation: PASS")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(onnx_path: Path, n_warmup: int = 10, n_runs: int = 100) -> float:
    """
    Run repeated inference and report mean / p95 latency.

    Uses batch_size=1 since that's what the real-time inference loop uses.
    """
    print(f"\nBenchmarking: {onnx_path.name}")
    print(f"  Using CPUExecutionProvider (simulating demo laptop)")

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    seq_len = CFG.model.seq_len
    crop_sz = CFG.preproc.crop_size
    dummy   = np.random.randn(1, seq_len, 3, crop_sz, crop_sz).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        sess.run(None, {"frames": dummy})

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"frames": dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = np.mean(latencies)
    p95_ms  = np.percentile(latencies, 95)
    print(f"  Mean latency : {mean_ms:.1f} ms")
    print(f"  P95  latency : {p95_ms:.1f} ms")
    print(f"  Equivalent FPS (single-frame throughput): {1000/mean_ms:.1f}")

    if mean_ms < 80:
        print("  [OK] Under 80 ms target — real-time capable on demo laptop")
    else:
        print("  [WARN] Over 80 ms — consider INT8 quantisation or smaller backbone")

    return mean_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=str, default=str(CKPT_DIR / "best.pt"),
                        help="Path to trained checkpoint")
    parser.add_argument("--out",       type=str, default=str(ONNX_PATH),
                        help="Output path for FP32 ONNX model")
    parser.add_argument("--out_int8",  type=str, default=str(ONNX_INT8),
                        help="Output path for INT8 quantised model")
    parser.add_argument("--quantize",  action="store_true",
                        help="Also export INT8 quantised version")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark FP32 and INT8 models on CPU")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path  = Path(args.out)
    int8_path = Path(args.out_int8)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    # Load model on CPU (ONNX export must be done from CPU)
    print(f"Loading checkpoint: {ckpt_path}")
    model = EngagementModel()
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', '?'):.4f})")

    # Export FP32
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(model, out_path)

    # Optional: quantise
    if args.quantize:
        quantize_model(out_path, int8_path)

    # Optional: benchmark
    if args.benchmark:
        benchmark(out_path)
        if args.quantize and int8_path.exists():
            benchmark(int8_path)

    print(f"\nDone. Copy to demo laptop:")
    print(f"  scp {out_path} user@laptop:~/clippy/")
    if args.quantize:
        print(f"  scp {int8_path} user@laptop:~/clippy/")


if __name__ == "__main__":
    main()
