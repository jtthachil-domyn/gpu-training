# 09 — Data Pipeline Checklist

This document contains the complete data pipeline checklist from the original text with detailed expansions.

---

# Tokenization (LOCAL)

- Tokenize dataset once, not at runtime.
- Use HF `tokenizers` for speed.
- Save tokenized dataset as `.arrow`, `.pt`, or `.jsonl`.
- Avoid tokenizing inside the data loader for performance.

### Notes:

- Tokenization is CPU-bound; local execution is enough.
- Vocabulary size must match model config.

---

# Dataset Sharding (LOCAL + GPU)

### LOCAL:

- Simulate per-rank sharding using:
  - DistributedSampler
  - Manual slicing (`indices[start:end]`)
- Validate that each rank gets unique shards.

### GPU:

- Same logic but executed across real GPUs.
- Must perform deterministic shuffling across nodes.

---

# Efficient Storage (GPU)

- Use WebDataset or tar-sharded datasets for high throughput.
- Use memory-mapped files to reduce RAM load.
- Use S3 or NFS as shared storage for multi-node workers.

---

# Reproducible Seed Handling (LOCAL + GPU)

Set seeds for:

- Python random
- NumPy
- PyTorch CPU
- PyTorch MPS (local)
- PyTorch CUDA (later)

Example:

```
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

Notes:

- Distributed training requires syncing seeds across workers.
- Ensure DistributedSampler uses `set_epoch()`.

---

# Data Loader Design (LOCAL)

- Small batch size locally.
- Use `num_workers=0` or `1` on Mac (due to MPS limitations).
- Use collation function for padding & masks.

---

# Data Loader Design (GPU)

- Use `num_workers=4–16`.
- Enable `pin_memory=True` for faster CPU→GPU transfer.
- Enable `prefetch_factor` depending on disk bandwidth.

---

# Validation Data

- Create explicit validation split.
- Evaluate at consistent intervals.
- Compute perplexity, loss curves, and next-token predictions.

---

This checklist ensures that data feeding is correct both locally and on GPU clusters.
