#!/bin/sh

rm -rf mlruns/

python3 train.py \
    --dataset-dir ../../../syndoku/dataset \
    --batch-size 4 \
    --epochs 2

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/studio-lab-user/.conda/envs/ml/lib
