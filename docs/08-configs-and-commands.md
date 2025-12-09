# 08 — Example Configs & Commands (LOCAL vs GPU)

This file includes ALL example configs and commands from the original document, expanded with notes.

---

# LOCAL — DDP Simulation Command

```
torchrun --nproc_per_node=4 train.py
--model-config config.json
--data data/
--batch-size 8
--epochs 10
```

### Notes:

- This uses CPU/MPS as backend.
- `nproc_per_node=4` simulates 4 rank workers.
- Each rank gets a separate mini-batch or shard of data.
- This teaches the *logic* of multi-GPU training without actual GPUs.
- On a real GPU cluster, backend will be NCCL; locally it will be Gloo.

---

# GPU — DeepSpeed Config Example

```
{
"train_batch_size": 32,
"gradient_accumulation_steps": 1,
"fp16": {"enabled": true},
"zero_optimization": {"stage": 2},
"optimizer": {"type":"AdamW", "params":{"lr": 1e-4}}
}
```

### Notes:

- `train_batch_size` is global batch size across all GPUs.
- ZeRO stage 2 partitions optimizer states across ranks.
- ZeRO stage 3 partitions weights, gradients, and optimizer states.
- Mixed precision FP16 significantly saves memory.
- This config is executable only on an NVIDIA GPU cluster.

Run:

```
deepspeed --num_gpus=4 train_deepspeed.py --deepspeed_config deepspeed_config.json
```

---

# GPU — SLURM sbatch Template

#!/bin/bash
#SBATCH --job-name=ds-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
module load cuda/11.8
srun deepspeed --num_gpus=4 train_deepspeed.py --deepspeed_config ds_cfg.json

Notes:

- `nodes=2` = multi-node job.
- SLURM handles resource allocation and worker placement.
- `srun` launches distributed workers across nodes.
- This will not run on Mac — only on SLURM-enabled GPU cluster.

---

# GPU — Dockerfile Example

```
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip git
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118RUN pip3 install deepspeed transformers datasets accelerate wandb
WORKDIR /workspace
COPY . /workspace
```


Notes:

- Must pin CUDA version.
- Ensures all nodes run identical environment.
- Simplifies reproducible distributed training.

---

# Additional Useful Commands

### LOCAL: Run training with profiling

```
python train.py --profile
```

### GPU: Inspect GPU state

```
nvidia-smi
nvidia-smi dmon
gpustat
```

### GPU: NCCL debugging

```
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
```

---

All commands above are preserved exactly, with additional explanations.
