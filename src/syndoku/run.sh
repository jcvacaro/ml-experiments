#!/bin/sh

rm -rf mlruns/

python3 train.py \
    --dataset-dir ../../../syndoku/dataset \
    --batch-size 4 \
    --max_epochs 10 \
    --lr 3e-5
