FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.8-py3

RUN apt-get update \
    && apt-get install -y \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

WORKDIR /home
COPY requirement.txt .
RUN pip3 install -r requirement.txt --no-cache-dir \
    && pip3 install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git@v0.3.0 \
    && rm *.txt

SHELL ["/bin/bash", "-c"]