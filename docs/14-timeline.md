# 14 — Sample Timeline (2–3 Weeks)

This preserves the original timeline and expands details for clarity.

---

# Week 0 — Preparation (LOCAL)
- Set up Python and PyTorch (MPS backend).
- Validate MPS supports required operations.
- Prepare dataset(s) and tokenization.
- Build repo structure.
- Prepare tiny transformer code.
- Validate `torchrun` local simulation.
- Prepare slides for Session A.

---

# Week 1 — Session A + Lab 1 (LOCAL)
### Focus:
- Core concepts
- Tiny model training
- Checkpointing
- Logging
- Debugging shapes & loops

### Outputs:
- First working model
- Training curves
- Understanding of model architecture

---

# Week 2 — Session B + Lab 2 (LOCAL)
### Focus:
- Simulated multi-GPU training
- torchrun multi-process execution
- distributed sampler
- sharding
- rank logging
- infrastructure concepts (GPU-only theory)

### Outputs:
- Multi-process simulation code
- Per-rank logs
- First distributed-style checkpoints

---

# Week 3 — Session C + Open Lab (LOCAL + GPU Theory)
### Focus:
- Multi-node training logic (explained)
- SLURM job structure (explained)
- DeepSpeed configs (explained)
- Advanced parallelism strategies
- Scaling models

### Outputs:
- Complete logical understanding
- GPU-ready configs
- Team prepared for real cluster execution

---

# Optional Week 4 — GPU Execution (when cluster access is provided)
- Run 100M parameter model on real GPU
- Use DeepSpeed ZeRO-2/3
- Use SLURM allocation
- Log results
- Debug real GPU issues

This timeline ensures smooth progression from teaching → simulation → real execution.
