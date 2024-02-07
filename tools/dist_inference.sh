#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
OUT_DIR=$4
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/inference.py $CONFIG $CHECKPOINT $OUT_DIR --launcher pytorch ${@:5}
