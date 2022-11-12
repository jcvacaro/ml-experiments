#!/bin/bash

DOCKER_IMAGE='guia-exp:0.3.1'

docker run \
    -e HOME=/root \
    -v $PWD/cache:/root \
    --gpus device=all \
    -it \
    --rm \
    --name=guia-charts-$USER \
    --ipc=host \
    --shm-size=8gb \
    -p 8888:8888 \
    -v $PWD:/workspace \
    -v $HOME/workspace/ICPR2020-UB-PMC:/data \
    -v $PWD/../guia_mlutils:/guia_mlutils \
    -e PYTHONPATH=/guia_mlutils \
    $DOCKER_IMAGE /bin/bash


