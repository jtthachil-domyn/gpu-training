# 03 — High-Level Approach (LOCAL vs GPU)

This section maps every part of the training program into what runs locally and what runs later on the GPU cluster. Nothing is removed or simplified; more detail is added.

---

# 1. Foundations (concepts) — LOCAL

Concepts we teach and simulate locally:

- Why distributed training is required
- Data parallelism (replicas of model across workers)
- Model parallelism (shard model layers or weights)
- Pipeline parallelism (microbatches across pipeline stages)
- Mixed precision training (FP16/BF16)
- Loss scaling
- Checkpointing strategies

We run:
- toy models
- tiny transformers
- local multi-process simulations

This gives the team foundational understanding.

---

# 2. Environments & Infra — GPU

These require real NVIDIA GPUs:

- CUDA drivers and compatibility matrices
- NVML for monitoring
- NCCL communication and tuning
- Docker images with GPU support
- Multi-node networking parameters
- SLURM job lifecycle
- Kubernetes GPU scheduling
- S3/NFS/Lustre shared storage

We will **teach** this now but **execute** it later.

---

# 3. Distributed Frameworks

### PyTorch DDP — LOCAL simulation
- Use `torchrun` to launch multiple processes.
- Teach `RANK`, `WORLD_SIZE`, `LOCAL_RANK`.
- Simulate gradient synchronization.
- Simulate distributed samplers.
- Demonstrate per-rank logs and checkpoints.

### Hugging Face Accelerate — LOCAL
- Device placement strategies.
- Accelerator abstraction.
- Same code will scale to GPU cluster with minimal changes.

### DeepSpeed — GPU
- ZeRO Stage 1/2/3.
- CPU/NVMe offloading.
- Partitioning optimizer states.
- Memory saving vs communication overhead.
- DeepSpeed launcher.

---

# 4. Practical Workflows — LOCAL
We can run all of these locally:

- Tokenization  
- Dataset preparation  
- Sharding  
- Reproducibility (seed control)  
- Logging (TensorBoard, WandB offline)
- Checkpoint save/load
- Evaluation scripts

The team learns full project workflows end-to-end.

---

# 5. Operations

### Local Only:
- Debugging training loops
- CPU/MPS profiling
- Logging gradients, losses, throughput
- Memory estimates of toy models

### GPU (later):
- `nvidia-smi` monitoring
- GPU memory fragmentation issues
- NCCL deadlocks
- Network latency bottlenecks
- Real large batch training

---

This high-level approach ensures a clean transition:

**LOCAL (build understanding) → GPU (execute real training)**
