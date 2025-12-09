# 13 — Deliverables (Complete List)

This file includes everything the team is expected to produce as part of the training program.  
Nothing has been abstracted; details are expanded.

---

# LOCAL Deliverables (Executable Now)

### 1. Slide Deck
- Distributed training concepts  
- LOCAL vs GPU distinctions  
- Parallelism types  
- DDP simulation diagrams  
- Workflow overview

### 2. GitHub Repository
Containing:
- Scripts
- Labs
- Configs
- Documentation (this folder)

### 3. Tiny Transformer Model
- Minimal transformer architecture (`nn.Module`)
- Positional embeddings
- Multi-head attention
- Feed-forward network
- Layer norms
- Final LM head

### 4. Training Loop
- forward pass  
- loss computation  
- backward pass  
- optimizer step  
- gradient clipping  
- metric tracking  

### 5. DDP Simulation
- using `torchrun --nproc_per_node=4`
- rank-aware logging  
- distributed sampler  
- sharded dataset  

### 6. Checkpoints
- model checkpoints  
- optimizer checkpoints  
- simulated rank-specific checkpoints  

### 7. Evaluation Script
- perplexity computation  
- validation dataset  
- sample text generation  

### 8. Inference Script
- load checkpoint  
- generate predictions  
- interactive inference  

---

# GPU Deliverables (Prepared Now, Executed Later)

### 1. DeepSpeed Config
- ZeRO stage  
- FP16  
- microbatch size  
- global batch size  

### 2. SLURM Script
- multi-node launcher  
- srun parameters  
- time allocation  
- environment setup  

### 3. Dockerfile
- CUDA base image  
- pinned PyTorch version  
- DeepSpeed install  
- training environment  

---

# Supplementary Deliverables

### Troubleshooting Cheat Sheet
- common LOCAL errors  
- common GPU/NCCL issues  
- debugging steps  
- logging patterns  

### Tutorial Recordings (Optional)
- walkthrough of Sessions A, B, C  
- demonstration of code workflows  

---

All deliverables prepare the team for a clean transition from LOCAL simulation to full GPU cluster execution.
