# GPU Training Curriculum — Tutorial Guide

This repository is designed to be a self-paced or instructor-led course on Distributed Training.

## Pedagogy Strategy
The core philosophy is **"Learn Locally, Scale Globally"**. We avoid the complexity of real clusters initially by simulating everything on a local machine (Mac M-series or Linux CPU).

### 1. The Learning Path
Students should follow the labs in strict order.

| Lab | Concept | Activity | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Lab 1** | Transformers & Training Loop | Build & Train a tiny model from scratch. | Understanding the `forward` -> `loss` -> `backward` loop without distributed complexity. |
| **Lab 2** | Data Parallelism (DDP) | Simulate 2+ GPUs on 1 CPU. | Understanding `rank`, `world_size`, gradient synchronization, and sharding. |
| **Lab 3** | Multi-Node Clusters | Simulate generic node networking. | Understanding `NODE_RANK`, `MASTER_ADDR`, and how clusters strictly orchestrate processes. |

---

## Lab 1: The Foundation
**Goal**: Verify the student can make a model learn *anything*.

1.  **Theory**: Explain that "Big" models are just "Tiny" models scaled up. The code logic is identical.
2.  **Code Walkthrough**:
    *   Show `TinyTransformer` class in `scripts/train_tiny_transformer.py`.
    *   Highlight the manual training loop (no Trainer abstraction yet).
3.  **Exercise**:
    *   Run: `python scripts/train_tiny_transformer.py --epochs 2`
    *   **Challenge**: Ask them to change `num_hidden_layers` in `config/config.json` to 4 and observe if loss drops faster or slower.

## Lab 2: The Distributed Shift
**Goal**: Demystify "Distributed". It's just running the same script N times.

1.  **Theory**: Explain DDP. "We copy the model N times. Each copy sees different data. They average gradients."
2.  **Code Walkthrough**:
    *   Show `dist.init_process_group`.
    *   Show `DistributedSampler` (Crucial: proof that data is split).
3.  **Exercise**:
    *   Run: `torchrun --nproc_per_node=2 scripts/ddp_simulation_train.py`
    *   **Observation**: Look at `logs/rank_0/train.log` vs `logs/rank_1/train.log`.
    *   **Question**: "Why are the loss values slightly different per step but similar?" (Ans: Different data batches).

## Lab 3: The Cluster Mental Model
**Goal**: Understand the environment *around* the code.

1.  **Theory**: On a real cluster, you don't type `torchrun`. You ask SLURM to do it. SLURM sets environment variables.
2.  **Code Walkthrough**:
    *   Ref `scripts/fake_multi_node_launcher.py`. Be explicit that this is a *simulation* of what SLURM does.
3.  **Exercise**:
    *   Run simulation with `export NODE_RANK=1 ...`.
    *   **Challenge**: "Pretend you are Node 3 of 4. What is your Global Rank range?"

## Next Steps (Real Hardware)
After Lab 3, the student is ready to read `config/slurm_example.sbatch` and understand it line-by-line, transitioning effectively to real GPU clusters.
