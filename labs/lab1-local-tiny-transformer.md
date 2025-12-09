# Lab 1 — Local Tiny Transformer Training (M4 Mac, MPS Backend)

This lab teaches you how to build and train a tiny transformer model **entirely on your local Mac**, with no GPUs, CUDA, or cluster.

Everything here is executable immediately.

---

## Objectives

- Build a tiny transformer model from scratch.
- Train it on CPU/MPS.
- Produce checkpoints.
- Evaluate the model.
- Understand the basics before moving to distributed training.

---

# Part 1 — Environment Setup (LOCAL)

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio transformers datasets tokenizers accelerate tensorboard wandb
```

Verify MPS availability:

```python
import torch
print(torch.backends.mps.is_available())
```

Expect: True.

# Part 2 — Implement Tiny Transformer (LOCAL)

Model should include:

- Token embedding
- Positional embedding
- 1–2 Transformer blocks
- Multi-head attention
- Feed-forward network
- LayerNorm
- Final LM head

Suggested tiny settings:

- hidden_size = 64 or 128
- num_heads = 2–4
- num_layers = 1 or 2
- sequence_length = 32–64

Purpose: Fast training and clear understanding.

# Part 3 — Training Loop (LOCAL)

Training steps:

- Load small dataset
- Tokenize
- Create dataloader
- Forward pass
- Compute loss
- Backward
- Optimizer step
- Log metrics

Example loop:

```python
for step, batch in enumerate(dataloader):
    optim.zero_grad()
    logits = model(batch["input_ids"])
    loss = criterion(
        logits.view(-1, vocab_size),
        batch["labels"].view(-1)
    )
    loss.backward()
    optim.step()
```

# Part 4 — Checkpointing (LOCAL)

Save:

```bash
checkpoints/step_100.pt
```

Code:

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optim.state_dict(),
    "step": step
}, f"checkpoints/step_{step}.pt")
```

Resume:

```python
state = torch.load(...)
model.load_state_dict(state["model"])
optim.load_state_dict(state["optimizer"])
```

# Part 5 — Evaluation (LOCAL)

Tasks:

- Compute perplexity
- Evaluate on validation set
- Generate sample predictions

# Part 6 — Visualization (LOCAL)

Start tensorboard:

```bash
tensorboard --logdir logs/
```

Track:

- loss
- LR schedule
- gradient norms

## Deliverables

- tiny transformer model file
- training loop
- checkpoints
- evaluation script
- logs/tensorboard graphs

This lab prepares you for multi-process simulation (Lab 2).
