#!/bin/sh

python3 train.py \
    --dataset-dir ../../../syndoku/dataset \
    --batch-size 4 \
    --epochs 2 \
    2>&1 | tee output.txt

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/studio-lab-user/.conda/envs/ml/lib
