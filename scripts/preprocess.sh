#!/usr/bin/env bash
export PYTHONPATH="$(dirname "$0")/..:$(dirname "$0")/../external:${PYTHONPATH:-}"

python "$(dirname "$0")/../preprocess/save_edge_mask.py" \
  "$(dirname "$0")/../configs/PRO/save_edge_mask.py" \
  --work-dir "$(dirname "$0")/../Datasets/UnrealStereo4K/BFM" \
  --process-num 1