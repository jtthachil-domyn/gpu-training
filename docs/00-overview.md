# 00 — Overview

This document provides a complete, detailed workflow for training our team on distributed training concepts using:

- LOCAL execution on the M4 MacBook Pro (simulation of distributed systems)
- GPU execution later on an NVIDIA-based cluster (MareNostrum or equivalent)

Nothing in this repository removes or abstracts the original content.  
All content from planning, teaching, and technical explanation is preserved exactly and expanded where necessary.

---

## What this documentation teaches

1. The conceptual foundations of distributed training.
2. How to simulate distributed training workflows locally on a Mac.
3. What real GPU cluster training looks like (DeepSpeed, NCCL, CUDA, SLURM).
4. How to structure an internal training program for the team.
5. How to prepare team members so that once GPU access is available, they can immediately run large-scale jobs.

---

## What can be executed NOW (LOCAL)

The M4 MacBook can run:

- Tiny transformer training
- Multi-process DDP simulation (`torchrun`)
- Evaluation, metrics, checkpoints
- Tokenization & dataset preparation
- LoRA fine-tuning
- Profiling and debugging
- Full training loop implementation

This is enough to teach the entire workflow end-to-end.

---

## What will be executed LATER (GPU)

On MareNostrum or any NVIDIA cluster, we will finally run:

- DeepSpeed ZeRO-2/3
- Mixed precision (AMP FP16/BF16)
- Multi-GPU DDP training
- Multi-node NCCL-backed training
- SLURM/Kubernetes scheduled jobs
- High-throughput distributed data loading
- Model/tensor/pipeline parallelism

---

## Why we use this phased approach

- LOCAL stage builds intuition and teaches architecture.
- GPU stage executes high-performance workloads once infra is provided.
- The same code works in both environments with minimal changes.

---

## Who this documentation is for

- ML engineers
- Developers new to distributed systems
- Engineers preparing to train foundation models or guard models
- Team members who need exposure to HPC-style workflows

---

This overview sets the stage for the detailed sections that follow.