#!/usr/bin/env bash
export PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH
export PYTHONPATH="$(dirname "$0")/../external":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

CONFIG=$1

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
python  "$(dirname "$0")/../tools/train_disp.py" "$CONFIG" ${@:2}