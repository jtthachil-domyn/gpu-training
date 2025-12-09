# 04 — Tutorial Curriculum (Sessions A, B, C)

This curriculum contains every detail of the original plan, expanded with notes. All content is preserved.

---

# Session A — Concepts & Local Development (90 mins)

**All LOCAL, fully runnable on M4 Mac.**

### Slides

- What is distributed training?
- Why models exceed single-device memory.
- Data vs model vs pipeline parallelism.
- Autograd and computation graph overview.
- Forward pass, backward pass, gradient update.
- Intro to training loops.

### Demo (LOCAL)

- Build and train a tiny transformer (few layers).
- Use MPS backend.
- Observe loss decreasing.
- Save checkpoint every N steps.
- Resume from checkpoint.
- Validate model on held-out samples.

### Lab 1 (LOCAL)

- Run the tiny model.
- Modify embedding size, FFN width, number of heads.
- Debug shape mismatches (common beginner error).
- Track loss curves over time.
- Visualize checkpoints.

---

# Session B — Single-Node Multi-GPU & Infrastructure (90 mins)

**Mixed environment: LOCAL simulation + GPU theoretical content.**

### Slides (GPU concepts)

- How NCCL performs all-reduce.
- What gradient buckets are.
- CUDA kernels vs fused kernels.
- AMP autocast & GradScaler.
- Gradient accumulation strategies.
- Effective batch size and memory trade-offs.

### Demo (LOCAL)

Simulate multi-GPU execution using:

```
torchrun --nproc_per_node=4 train.py
```

Demonstrate:

- each process acting as a “GPU worker,”
- rank-specific logs,
- dataset sharding per rank,
- simulated gradient synchronization.

### Lab 2 (LOCAL)

- Modify world size and observe effects.
- Add rank-specific metrics.
- Force failure in rank 2 to simulate error conditions.
- Test distributed sampler behavior.

---

# Session C — Multi-node Distributed & Production Practices (90 mins)

**GPU-only concepts, simulated locally.**

### Slides (GPU)

- Multi-node NCCL: master address, master port.
- Environment variables: `WORLD_SIZE`, `NODE_RANK`, `RANK`, `LOCAL_RANK`.
- SLURM job structure:
  - sbatch → allocate → srun workers → cleanup.
- Kubernetes GPU pods and device plugins.
- Checkpointing across nodes.
- Distributed logging aggregation.

### Demo (LOCAL)

- Fake a 2-node job by manually assigning ranks.
- Simulate distributed barriers.
- Show SLURM config and DeepSpeed config.
- Walk through how cluster nodes coordinate.

### Lab 3 (LOCAL)

- Build a simulated multi-node script.
- Save “ranked checkpoints.”
- Evaluate tiny model.
- Discuss scaling constraints.

---

# Learning Outcome

After Sessions A, B, and C, every engineer will understand:

- how distributed training works,
- how to write code that runs in distributed mode,
- how cluster job scripts work,
- how to debug distributed issues,
- how to transition from LOCAL → GPU cluster with minimal friction.
