# 05 — Hands-On Mini Project

This mini project is **fully executable locally** and mirrors the structure of a true multi-GPU training workflow.
GPU components are explained and prepared but not executed yet.

---

# Project Goal

Train a **tiny transformer** model from scratch using:

- LOCAL training loop
- LOCAL DDP simulation (`torchrun`)
- LOCAL evaluation, metrics, checkpoints
- LOCAL inference script

GPU concepts (DeepSpeed, ZeRO, SLURM, NCCL) will be prepared as configs but executed later.

---

# Local Implementation Tasks

### 1. Build tiny transformer

- embedding layer
- positional embeddings
- transformer blocks (attention + FFN)
- layernorm
- final LM head

### 2. Train model using MPS

- small batch sizes
- gradient accumulation
- checkpoint every N steps
- log training loss

### 3. Add DDP simulation

Using:

```
torchrun --nproc_per_node=4 train.py
```

Include:

- per-rank logs
- distributed sampler
- rank-aware checkpoint paths
- correct world_size logic

### 4. Evaluate

- compute perplexity
- validate on a held-out dataset
- generate a few predictions

### 5. Inference script

- load final checkpoint
- run inference on new text
- print next-token predictions

---

# GPU Preparation Tasks (NOT RUN YET)

### Prepare DeepSpeed config

- fp16 enabled
- ZeRO stage 2 or 3
- train batch size definition
- partitioning strategies

### Prepare SLURM script

- job name
- nodes
- GPUs per node
- time allocation
- srun launcher

### Prepare Dockerfile

- CUDA 11.8 base
- PyTorch installation
- DeepSpeed installation
- working directory

---

# Expected Deliverables

- tiny model code
- training loop
- evaluation script
- distributed simulation code
- logs and checkpoints
- DeepSpeed config
- slurm script
- Dockerfile
- final inference script

This mini project ensures the team has **complete functional understanding** before touching real GPUs.
