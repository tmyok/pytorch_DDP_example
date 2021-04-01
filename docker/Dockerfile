# docker build --no-cache -t tmyok/pytorch181-cudnn8-cuda1122-ubuntu2004:v210402 .
ARG cuda_version=11.2.2
ARG cudnn_version=8
ARG ubuntu=20.04
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-runtime-ubuntu${ubuntu}

ENV DEBIAN_FRONTEND noninteractive

RUN apt -y update && apt -y upgrade && \
    apt -y install --no-install-recommends \
        python3-dev \  
        python3-numpy \
        python3-pip && \
    apt -y clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install tqdm==4.59.0