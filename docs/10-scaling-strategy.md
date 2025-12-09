# 10 — Scaling Strategy & Model-Parallel Notes

This file expands the original scaling guidelines while preserving all original content.

---

# LOCAL Scaling Strategy

### Tiny Models
- Train models <5M parameters.
- Use small embedding sizes.
- Reduce number of layers (1–2).
- Use gradient accumulation to simulate larger batches.
- Lower sequence length to reduce memory impact.

### Simulated Multi-GPU Logic
- Launch multiple processes with `torchrun`.
- Each process acts as a “fake GPU worker.”
- Demonstrates world_size, ranks, gradient sync logic.
- Enables debugging distributed architecture locally.

---

# GPU Scaling Strategy (executed later)

### Single-Node Multi-GPU
- Use PyTorch DDP backend (NCCL).
- Increase global batch size.
- Use FP16 or BF16 mixed precision.
- Use DeepSpeed ZeRO Stage 1 or 2.
- Monitor GPU utilization and memory fragmentation.

### Multi-Node Multi-GPU
- Split workers across nodes.
- Use NCCL over Ethernet or InfiniBand.
- Requires SLURM job scripts for worker orchestration.
- Check network bandwidth/latency.
- Cluster scheduler decides node placement.

---

# When to Use Model Parallelism

### Tensor Parallelism
- Split weights across GPUs.
- Required for >1B parameter models.
- Supported by Megatron-LM and DeepSpeed-MP.

### Pipeline Parallelism
- Split layers across GPUs.
- Requires microbatching.
- Good for deep networks with sequential structure.

---

# DeepSpeed ZeRO (GPU-only execution)

### Stage 1 — Optimizer State Sharding
- partitions optimizer states across ranks

### Stage 2 — Gradient + Optimizer State Sharding
- reduces memory load further

### Stage 3 — Parameters + Gradients + Optimizer Partitioning
- maximum memory savings  
- required for large models

---

# Checkpointing Strategy

### LOCAL
- Save checkpoints for tiny model.
- Simulate sharded checkpoints by saving rank-specific files.

### GPU
- Save complete model (fp32 master weights).
- Save optimizer states.
- Save random number generator states.
- Save sharded ZeRO checkpoints (multiple files per rank).

---

This scaling document prepares the team for both simulation-based work and real cluster execution.
