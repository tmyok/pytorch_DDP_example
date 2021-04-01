#!/bin/bash
# -*- coding: utf-8 -*-

docker run \
    -it \
    --rm \
    --gpus all \
    --shm-size=16g \
    --name PyTorch_DDP_example \
    --volume $(pwd)/../:/home/work/ \
    --workdir /home/work/ \
    tmyok/pytorch181-cudnn8-cuda1122-ubuntu2004:v210402