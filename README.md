# Distributed Training Tutorial — Full Documentation (LOCAL + GPU)

This repository contains the **complete training curriculum**, **detailed explanations**,and **simulation-based distributed training workflow** built to train the team using:

- **LOCAL** M4 Mac (simulation of multi-GPU concepts)
- **GPU** cluster (MareNostrum or similar NVIDIA hardware) — *execution later*


*All sections include full detail, with additional LOCAL vs GPU annotations.*

---

# Table of Contents

### Overview & Foundations

- [00 — Overview](docs/00-overview.md)
- [01 — ELI5 Intro for Leadership](docs/01-eli5-intro.md)
- [02 — Colab/Kaggle vs GPU Cluster Coding](docs/02-colab-vs-cluster.md)
- [03 — High-Level Approach (LOCAL vs GPU)](docs/03-high-level-approach.md)

### Full Curriculum (Sessions A, B, C)

- [04 — Tutorial Curriculum (All Sessions)](docs/04-tutorial-curriculum.md)
- [05 — Mini Project Deliverable](docs/05-mini-project.md)

### Requirements & Setup

- [06 — Prerequisites](docs/06-prerequisites.md)
- [07 — Software Stack (LOCAL + GPU)](docs/07-software-stack.md)

### Commands, Configs, Workflows

- [08 — Example Configs &amp; Commands](docs/08-configs-and-commands.md)
- [09 — Data Pipeline Checklist](docs/09-data-pipeline.md)
- [10 — Scaling Strategy Notes](docs/10-scaling-strategy.md)
- [11 — Monitoring &amp; Debugging](docs/11-monitoring-debugging.md)

### Operations & Planning

- [12 — Cost &amp; Compute Sizing](docs/12-cost-sizing.md)
- [13 — Deliverables &amp; Repo Outputs](docs/13-deliverables.md)
- [14 — Sample Timeline](docs/14-timeline.md)
- [15 — CUDA Clarification](docs/15-cuda-note.md)

---

# Labs (Hands-on Exercises)

- [Lab 1 — Local Tiny Transformer](labs/lab1-local-tiny-transformer.md)
- [Lab 2 — DDP Simulation on Local Machine](labs/lab2-ddp-simulation.md)
- [Lab 3 — Multi-node Logic Simulation](labs/lab3-multi-node-simulation.md)

---

# Scripts

These are placeholder files you will later populate using Cursor/Code/Claude.

See [`scripts/README.md`](scripts/README.md).

---

# Config Examples

- DeepSpeed config
- SLURM script
- Dockerfile

See [`configs/`](configs/).
