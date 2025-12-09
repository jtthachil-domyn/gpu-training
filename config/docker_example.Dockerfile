FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install deepspeed transformers datasets accelerate wandb

WORKDIR /workspace
COPY . /workspace