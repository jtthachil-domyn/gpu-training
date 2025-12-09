# 06 — Prerequisites for Attendees

This document lists *all* prerequisites required for engineers to participate effectively in the distributed training tutorial.  
Nothing has been abstracted or removed—details have been expanded.

---

# Core Knowledge (All achievable using LOCAL machine)

### 1. Python Fundamentals
Attendees must be comfortable with:
- writing Python functions and classes
- importing modules
- using virtual environments
- using pip or conda
- reading stack traces

### 2. PyTorch Basics
Must understand:
- defining `nn.Module` classes
- forward pass
- parameters and optimizers
- training loop structure:
  - forward → loss → backward → optimizer step
- basic debugging of tensor shapes

We will teach deeper concepts (autograd graph, distributed training wrappers), but foundational PyTorch is required upfront.

### 3. Linux Command Line
Attendees must understand:
- navigating directories
- running Python scripts
- understanding shell arguments
- basic `grep`, `cat`, `tail -f`
- SSH (later when GPU access is available)
- file permissions (optional but helpful)

### 4. Git & Version Control
Needed for:
- cloning the tutorial repo
- pushing labs
- maintaining code quality
- reading diffs and committing code

### 5. Ability to Run Code Locally (M4 Mac or similar)
Attendees must be able to:
- install PyTorch with MPS backend
- run `torchrun` locally
- install packages like transformers, accelerate, datasets
- open Jupyter notebooks or VSCode

This ensures the entire tutorial can be completed without any GPU hardware.

---

# Optional (Recommended for GPU stage)
These are not required now but will greatly help later:

### 6. Familiarity with Docker
Understanding Dockerfiles is useful because:
- GPU clusters typically use containerized environments
- DeepSpeed jobs are usually launched inside containers
- Reproducibility depends on pinned environments

### 7. Basic HPC Concepts
Not required for Session A or B, but helpful for Session C:
- nodes vs GPUs vs tasks
- job queues
- SLURM allocation
- distributed storage

We will explain these concepts in detail.

---

# No CUDA knowledge required (for LOCAL phase)

Because everything in Sessions A & B is executed on MPS with CPU/MPS simulation:
- CUDA is NOT required now
- NCCL is NOT required now
- NVML/GPU flags are only explained conceptually

Knowledge of these will matter *only* once we run distributed jobs on MareNostrum GPUs.

---

# Summary of What Attendees Need Before Session A
- Python & PyTorch basics ✔
- Ability to run training locally ✔
- Ability to modify training loops ✔
- Ability to read/write simple scripts ✔
- Basic terminal usage ✔
