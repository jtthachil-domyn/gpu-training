# 12 — Cost, Compute & Sizing (Conceptual Only)

This section includes the complete sizing explanation from the original plan, expanded with additional detail.  
We do NOT run GPUs now; this is for planning and decision-making.

---

# Model Size vs Compute Needs

### Small Models (50M–200M parameters)
- Suitable for 4–8 GPU setups.
- Training time roughly: hours → low-cost.
- Good for experimentation and guard models.
- Batch sizes small to medium (32–256 tokens/batch).
- DeepSpeed optional but useful.

### Medium Models (1B parameters)
- Require 8–16 GPUs.
- Memory footprint large even with mixed precision.
- DeepSpeed ZeRO Stage 2/3 required.
- Activation checkpointing mandatory for long sequences.
- Training time: multiple days.

### Large Models (10B+ parameters)
- Must use:
  - tensor parallelism
  - pipeline parallelism
  - ZeRO stage 3
- Requires multi-node cluster.
- Requires high-bandwidth interconnect (InfiniBand preferred).
- Training cost and time significantly higher.

---

# Compute Considerations

### Batch Size Scaling
- Larger batches increase throughput but risk instability.
- Gradient accumulation increases effective batch size without requiring more memory.

### Sequence Length
- Training cost scales quadratically with sequence length.
- Longer sequences drastically increase activation memory.

### Optimizer Choices
- AdamW is standard but memory-heavy.
- Adafactor reduces memory but may reduce stability.
- Fused optimizers provide speedups on GPU.

---

# Storage Requirements
- Tokenized dataset storage.
- Checkpoint storage:
  - FP32 master weights
  - optimizer states
  - RNG states
  - distributed shards
- Logs + metrics + artifacts.

---

# Why Cost Sizing Is Included in Training
Even though we do not run GPU jobs now, understanding cost is crucial for:

- planning cluster usage
- estimating job time
- selecting model size
- budgeting GPU hours
- avoiding inefficient runs

This ensures that once GPU access becomes available, the team runs models efficiently and avoids unnecessary compute waste.
