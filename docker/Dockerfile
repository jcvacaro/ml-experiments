FROM paperspace/gradient-base:pt112-tf29-jax0317-py39-20230125

# Ubuntu packages
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
        build-essential \
        ca-certificates \
        ghostscript \
        locales \
        nano \
        poppler-utils \
        texlive \
        texlive-fonts-extra \
        texlive-latex-extra \
        texlive-luatex \
        texlive-pictures \
    && rm -rf /var/lib/apt/lists/*

# Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the locale: https://stackoverflow.com/questions/28405902/how-to-set-the-locale-inside-a-debian-ubuntu-docker-container
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

# pip packages
RUN pip3 install \
    accelerate \
    evaluate \
    jsonlines \
    kaggle \
    mlflow \
    optuna \
    pycocotools \
    torchmetrics
