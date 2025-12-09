# 02 — Colab/Kaggle vs GPU Cluster Coding

This section explains the difference between lightweight prototyping (Colab/Kaggle) and real distributed GPU cluster training. Nothing is removed or abstracted—details are expanded.

---

# Colab / Kaggle (LOCAL / limited environments)

### Characteristics
- Single VM.
- 1–4 GPUs (but often just 1).
- Sessions expire.
- Limited memory (12–24 GB typical).
- Great for prototyping and debugging.
- Good for unit testing model code.
- Good for validating training loops and tokenizers.
- Good for data inspection and preprocessing.

### Limitations
- Cannot simulate inter-GPU communication.
- Cannot simulate NCCL topologies (ring, tree, hierarchical).
- Cannot launch real multi-node distributed jobs.
- Cannot support large batch sizes or long sequences.
- Training speed is not representative of real clusters.

### Summary
Colab/Kaggle are **perfect for concept teaching, prototyping, and early development**, but **not** real distributed training.

---

# GPU Cluster Coding (REAL GPU execution)

### Characteristics
- Many GPUs across many machines.
- High-bandwidth links: 100Gb Ethernet or InfiniBand.
- Distributed libraries: NCCL, MPI, DeepSpeed, PyTorch DDP.
- Shared storage: NFS, Lustre, S3, etc.
- Job orchestration: SLURM, Kubernetes.
- Proper logging, monitoring, device placement.

### Required Knowledge
- CUDA toolchain
- NCCL communication configuration
- How to shard datasets across nodes/GPUs
- DeepSpeed ZeRO memory sharding
- Mixed precision (FP16/BF16)
- Checkpointing resiliency
- Node failures and recovery strategies

### Summary
Cluster coding is production-grade engineering involving both ML and distributed systems.

---

# Why we teach cluster coding BEFORE getting access

We will simulate everything locally and prepare all code, configs, and workflows so the moment we get GPU access, training can start immediately.

This reduces:
- onboarding time,
- debugging time,
- cost of trial-and-error on real GPU clusters.
