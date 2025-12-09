# 07 — Software Stack (LOCAL + GPU)

This document breaks down the exact software stack required for both:

1. LOCAL teaching and simulation (M4 Mac)
2. REAL GPU execution (NVIDIA cluster)


# LOCAL Software Stack (M4 Mac)

### Python

- Python 3.11+ recommended
- Use `pyenv` or `miniforge` for clean environments
- Virtual environment per tutorial session

### PyTorch (MPS backend)

Install via:

```
pip install torch torchvision torchaudio
```

Notes:

- MPS backend offers GPU-like acceleration via Apple Metal.
- Some ops may fall back to CPU—acceptable for teaching.
- AMP (autocast) is supported but slightly different from CUDA AMP.

### Hugging Face Libraries

Required packages:

```
pip install transformers datasets tokenizers accelerate
```

### Logging / Visualization

```
pip install tensorboard wandb
```

Notes:

- WandB will run in offline mode unless API key is provided.

### Other Local Tools

```
pip install psutil rich
```

Used for:

- monitoring CPU/MPS usage
- printing colored logs
- profiling training loops

---

# GPU Software Stack (NVIDIA Cluster — executed later)

### OS + Drivers

- Ubuntu 22.04 LTS recommended
- NVIDIA driver ≥ 525
- CUDA 11.8 or 12.x (must match PyTorch & DeepSpeed build)

### NCCL

- Required for GPU-to-GPU communication
- Must be installed with CUDA
- Needs tuning:
  - `NCCL_SOCKET_IFNAME`
  - `NCCL_IB_DISABLE`
  - `NCCL_DEBUG=INFO`

### PyTorch (CUDA build)

Install using:

```
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### DeepSpeed

Install after CUDA toolkit:

```
pip install deepspeed
```

Components:

- ZeRO optimizer
- CPU/NVMe offloading
- Fused kernels (Apex-like)

### Logging Tools

- WandB (online mode)
- TensorBoard with shared log directory
- NVML tools (`nvidia-smi`, `nvtop`, `gpustat`)

### Cluster Tools

- SLURM
- Kubernetes + NVIDIA device plugin (optional)
- Docker + nvidia-container-runtime
- S3 CLI (for checkpoints)
- systemd services for node initialization

---

# Image/Container Workflow (GPU)

GPU jobs typically run inside Docker containers:

- ensures reproducibility
- isolates CUDA versions
- stabilizes training environment
- simplifies cluster deployment

Base image example:

```
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

---

# Summary Table

| Component | LOCAL (Mac) | GPU Cluster | Notes                          |
| --------- | ----------- | ----------- | ------------------------------ |
| Python    | ✔          | ✔          | Same version recommended       |
| PyTorch   | MPS build   | CUDA build  | Different backends             |
| DeepSpeed | ✖          | ✔          | GPU-only                       |
| NCCL      | ✖          | ✔          | Required for DDP/ZeRO          |
| SLURM     | ✖          | ✔          | Production scheduling          |
| wandb     | offline     | online      | Same APIs                      |
| Docker    | optional    | required    | Cluster runs inside containers |
