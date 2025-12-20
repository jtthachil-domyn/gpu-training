# MN5 Quick Reference Card

> **One-page cheat sheet for MareNostrum 5 (Domyn Guard Team)**

---

## 🔐 Access Credentials

| Item | Value |
|------|-------|
| **Username** | `domy667574` |
| **Account** | `ehpc475` |
| **Password** | 1Password → "Login Mare Nostrum" |

---

## 🖥️ SSH Commands

```bash
# GPU Login (for development)
ssh domy667574@alogin1.bsc.es

# CPU Login (general work)
ssh domy667574@glogin1.bsc.es

# Data Transfer (large files ONLY)
ssh domy667574@transfer1.bsc.es
```

---

## 📊 Available Queues

| Queue | Max Time | Max Cores | Use For |
|-------|----------|-----------|---------|
| `acc_debug` | 2h | 640 | Quick tests |
| `acc_ehpc` ⭐ | 72h | 8000 | Production |
| `acc_interactive` | 2h | 40 | Interactive |

```bash
# Check your queues
bsc_queues

# Check job queue
sq  # or squeue -u $USER
```

---

## 🚀 Quick Start Commands

### Interactive Session (Debug)
```bash
salloc --account=ehpc475 --qos=acc_debug \
  --partition=acc --gres=gpu:1 \
  --cpus-per-task=20 --time=01:00:00
```

### Submit Batch Job
```bash
sbatch my_job.sbatch
```

### Check Job Status
```bash
sq                          # Your running jobs
sacct -j <JOBID>           # Job history
scancel <JOBID>            # Cancel job
```

---

## 📁 Storage Locations

| Path | Speed | Lifespan | Use For |
|------|-------|----------|---------|
| `$HOME` | Slow | Permanent | Config, scripts |
| `/gpfs/projects/ehpc475` | Medium | Permanent | Code, models |
| `/gpfs/scratch/ehpc475` | **Fast** | 2 weeks | Training data |

```bash
# Check quota
bsc_quota
```

---

## 🐳 Singularity (No Docker!)

```bash
# Pull container
apptainer pull pytorch.sif docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Run with GPU
apptainer exec --nv -B /gpfs/projects/ehpc475:/app ./pytorch.sif python train.py
```

---

## ⚠️ Critical Rules

1. **NO internet** on compute nodes — prepare everything offline
2. **20 CPUs per 1 GPU** — violating this = infinite queue wait
3. **Don't run heavy work on login nodes** — auto-killed after 10 min
4. **Use transfer nodes for data** — `transfer1.bsc.es`
5. **Scratch is wiped every 2 weeks** — backup important files!

---

## 🔧 Useful Commands

```bash
bsc_acct               # Check Compute Budget & Usage
bsc_queues             # Available queues
bsc_quota              # Storage quota
module av              # Available software
module load python     # Load Python
nvidia-smi             # GPU status (on compute node)
```

---

## 📚 Resources

- [MN5 Docs](https://www.bsc.es/supportkc/docs/MareNostrum5/intro)
- [SLURM Guide](https://www.bsc.es/supportkc/docs/MareNostrum5/slurm)
- [Singularity Docs](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html)
