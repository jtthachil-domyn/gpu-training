# 11 — Monitoring, Logging & Debugging (LOCAL vs GPU)

This document expands the operational aspects of training.

# LOCAL Monitoring & Debugging (M4 Mac)

### CPU/MPS Profiling

- Track time per batch.
- Track CPU utilization via `psutil`.
- Use PyTorch’s autograd profiler:

```
with torch.autograd.profiler.profile(use_cuda=False) as prof:
output = model(input)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

### Logging Tools

- TensorBoard:

```
tensorboard --logdir logs/
```

- WandB in offline mode for metric tracking:

```
WANDB_MODE=offline wandb init
```

### Gradient & Model Debugging

- Print gradient norms per layer.
- Check for exploding/vanishing gradients.
- Validate model parameters using:

```
for name, p in model.named_parameters():
print(name, p.shape)
```

- Inspect model memory footprint (estimated) using:

```
sum(p.numel() for p in model.parameters()) * 4 / 1e6 # MB for FP32
```

### Distributed Simulation Debugging

- Log rank ID, world size, process group initialization.
- Validate per-rank data sharding.
- Ensure rank 0 performs aggregation tasks.

---

# GPU Monitoring & Debugging (Executed later on real cluster)

### NVIDIA Tools

- `nvidia-smi`: view GPU memory, utilization.
- `nvidia-smi dmon`: per-GPU stats over time.
- `gpustat`: quick GPU summary across nodes.

### NCCL Debugging

Enable debugs:

```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0
```

Common issues detected:

- asynchronous timeouts
- connectivity failures
- mismatched ranks
- deadlocks during all-reduce

### PyTorch DDP Debugging

Common flags:

```
TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### Memory Debugging

- Monitor fragmentation using PyTorch memory snapshots.
- Use `torch.cuda.memory_summary()`.

### DeepSpeed Debugging

- Inspect ZeRO partition logs.
- Validate optimizer state partitioning.
- Log CPU/NVMe offloading if enabled.

---

# Logging Guidelines (LOCAL + GPU)

### LOCAL

- Write per-rank logs to local folders: `logs/rank_0`, `logs/rank_1`, etc.
- Use small text-based logs for clarity.
- Visualize using TensorBoard.

### GPU

- Use shared filesystem logs.
- Enable WandB for distributed logging.
- Ensure that logs do not overwrite across ranks:
  - use unique run IDs
  - use `group=job_name`

---

# Debugging Common Failures

### LOCAL Failures

- Shape mismatches
- Incorrect tokenization lengths
- Wrong masking logic
- Training loop errors
- Gradient not flowing (debug with `.retain_grad()`)

### GPU Failures

- NCCL hangs
- Rank mismatch
- SLURM environment not passing correct variables
- Out-of-memory (OOM) due to large batch or sequence length
- Failed gradient synchronization

---

Monitoring & debugging is one of the most important topics because distributed training often fails due to environment misconfiguration, not model code.
This document ensures the team understands both LOCAL and GPU-side techniques.
