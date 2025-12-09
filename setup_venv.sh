#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio transformers datasets tokenizers accelerate tensorboard wandb
