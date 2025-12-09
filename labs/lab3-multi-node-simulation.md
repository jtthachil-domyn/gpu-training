# Lab 3 — Multi-Node Distributed Training Simulation (LOCAL)

This lab simulates a **multi-node cluster environment** using environment variables and local processes.

You will NOT need:
- any GPUs  
- any cluster  
- SLURM access  
- CUDA  

But you will understand how **real multi-node jobs** work.

---

## Objectives

- Understand node rank vs global rank  
- Understand master address + port  
- Understand how clusters launch workers  
- Simulate multi-node rank mapping  
- Simulate barriers and coordination  
- Inspect a SLURM script logically  

---

# Part 1 — Simulating Node Environment Variables

Simulate:

```bash
export NODE_RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=23456
```

Then run:

```bash
python fake_multi_node_launcher.py
```

# Part 2 — Simulate Two Nodes Locally

Node breakdown:

- Node 0 → ranks 0,1,2,3
- Node 1 → ranks 4,5,6,7

Mapping logic:

```python
global_rank = node_rank * gpus_per_node + local_rank
```

Print in logs:

- node_rank
- local_rank
- global_rank
- world_size

# Part 3 — Simulating Distributed Barriers

```python
dist.barrier()
```

On a real cluster:

- blocks until all processes reach this point
- prevents race conditions

Locally, you simulate this by printing barrier points or using multiprocess rendezvous.

# Part 4 — SLURM Multi-Node Workflow (Concept Only)

Explain how SLURM sets:

- SLURM_PROCID
- SLURM_NODEID
- SLURM_LOCALID
- SLURM_NTASKS

And how srun launches training workers across nodes.

# Part 5 — DeepSpeed Multi-Node Concepts (Concept Only)

Explain:

- partitioning optimizer states across nodes
- sharding gradients across nodes
- ZeRO-3 partitioning
- cross-node communication costs

No GPU used.
No real DeepSpeed execution yet.

## Deliverables

- fake_multi_node_launcher.py
- logs showing node rank + global rank
- simulated multi-node barriers
- understanding of cluster worker orchestration

After this lab, the team fully understands single-node + multi-node distributed workflows before using any real GPU cluster.
