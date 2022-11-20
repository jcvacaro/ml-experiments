FROM python:3.10-slim

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
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR "/workspace"
