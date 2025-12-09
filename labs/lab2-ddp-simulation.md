# Lab 2 — Distributed Data Parallel Simulation (LOCAL using torchrun)

This lab simulates **Distributed Data Parallel (DDP)** training on a single machine using multiple processes.

No GPU cluster needed.  
No CUDA.  
Everything runs with CPU/MPS + Gloo backend.

---

## Objectives

- Understand RANK, WORLD_SIZE, LOCAL_RANK  
- Use torchrun to launch multiple workers  
- Use DistributedSampler  
- Perform per-rank logging  
- Create distributed-style checkpoints  

---

# Part 1 — Launching DDP Simulation

Run:

```bash
torchrun --nproc_per_node=4 ddp_simulation_train.py
```

This launches 4 processes, each acting like a GPU worker (even though all run locally).

# Part 2 — Initialize Process Group

Inside your script:

```python
dist.init_process_group(
    backend="gloo",
    rank=rank,
    world_size=world_size
)
```

Notes:

- LOCAL uses gloo
- GPU cluster will use nccl

# Part 3 — Rank-Aware Logging

Each process writes logs to:

```bash
logs/rank_0/train.log
logs/rank_1/train.log
logs/rank_2/train.log
logs/rank_3/train.log
```

Log:

- rank
- world_size
- loss
- checkpoint info
- batch indices

# Part 4 — Dataset Sharding (DistributedSampler)

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)
```

Ensures:

- each rank sees unique data
- deterministic epoch-based shuffling

# Part 5 — Backward + Gradient Logic

Even though CPU/MPS backend is used, DDP still:

- synchronizes gradients
- averages them across workers
- ensures model replicas remain identical

# Part 6 — Distributed Checkpointing

Either save per-rank:

```
checkpoint_rank_0.pt
checkpoint_rank_1.pt
...
```

Or consolidate on rank 0:

```python
if rank == 0:
    torch.save(...)
```

## Deliverables

- ddp_simulation_train.py
- rank-specific logs
- distributed sampler usage
- correct world_size/rank setup
- simulated distributed training checkpoints

This lab prepares you for multi-node logic (Lab 3).
