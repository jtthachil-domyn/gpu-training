
---

# 📁 **scripts/README.md**
```md
# Scripts Overview

This folder contains all training-related scripts.  
Currently, these are placeholders that you will later populate using Cursor, Claude, Codex, or ChatGPT Code.

---

# Files

## train_tiny_transformer.py
- Implements tiny transformer model
- Training loop
- Checkpoint logic
- Evaluation logic
- Runs on LOCAL machine (MPS/CPU)

## ddp_simulation_train.py
- Multi-process distributed simulation
- Rank logging
- Dataset sharding
- DistributedSampler usage
- Simulated gradient synchronization

## fake_multi_node_launcher.py
- Simulates multi-node distributed environment
- Defines node rank, local rank, master address
- Appears similar to SLURM/Kubernetes worker launch
- Handles environment variables and rank mapping

---

# Notes
- All scripts are LOCAL and executable without GPUs.
- Actual GPU versions will be written later when cluster access is available.
- This structure mirrors real-world distributed ML repos.