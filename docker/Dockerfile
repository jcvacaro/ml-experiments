FROM python:3.9-slim

# Ubuntu packages
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        locales \
        nano \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Set the locale: https://stackoverflow.com/questions/28405902/how-to-set-the-locale-inside-a-debian-ubuntu-docker-container
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

# pip packages
RUN pip3 install \
    albumentations \
    jupyter \
    optuna \
    mlflow \
    pycocotools \
    pytorch-lightning \
    torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR "/workspace"
