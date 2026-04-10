#!/bin/bash
# Wrist-camera object attention correlation pipeline.
# Reads pre-computed attention H5 files from RESULTS_ROOT (produced by pipeline.py)
# and correlates them with wrist object masks from perception_pipeline.py.
# Results are written alongside the existing H5 files in RESULTS_ROOT.

CAMERA="left"
DATASET="faraz"
DATA_ROOT="/mnt/sda/edward/projects/toy_cube_benchmark/${DATASET}"
RESULTS_ROOT="/mnt/sda/edward/projects/pi05_vis/${DATASET}/${CAMERA}"

echo "DATA_ROOT:    ${DATA_ROOT}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo ""

set -e

uv run python viz/h1_wrist_object_corr.py \
    "$DATA_ROOT" \
    "$RESULTS_ROOT" \
    --from-h5 --force

# To re-run from scratch (clears existing markers):
#   bash viz/h1_wrist_object_corr.sh --force
# To re-generate plots only (no processing):
#   uv run python viz/h1_wrist_object_corr.py "$DATA_ROOT" "$RESULTS_ROOT" --aggregate-only
