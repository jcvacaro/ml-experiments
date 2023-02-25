#!/bin/sh

python3 train.py \
    --dataset-dir $HOME/data/syndoku \
    --batch-size 4 \
    --epochs 5 \
    2>&1 | tee output.txt

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/studio-lab-user/.conda/envs/ml/lib
