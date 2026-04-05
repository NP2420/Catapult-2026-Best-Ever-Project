#!/bin/bash
# ============================================================
# slurm/debug.sh
# Start an interactive session on a GPU node for debugging.
# Use 'standby' QOS — free but 4-hour max.
#
# Run: bash slurm/debug.sh
# ============================================================

srun \
    --partition=ai \
    --account=catapult \
    --qos=standby \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=28 \
    --gres=gpu:1 \
    --mem=100G \
    --time=04:00:00 \
    --pty \
    bash
