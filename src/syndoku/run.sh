#!/bin/sh

DOCKER_IMAGE='ml-exp:01'

docker run \
    -it \
    --rm \
    --ipc=host \
    --shm-size=4gb \
    -p 8888:8888 \
    -v $PWD/../..:/workspace \
    -v $HOME/Workspace/jcvacaro/syndoku/dataset:/data \
    $DOCKER_IMAGE /bin/bash
