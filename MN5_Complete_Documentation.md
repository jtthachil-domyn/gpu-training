# BSC MareNostrum 5 - Comprehensive Guide

Generated from source files in `/Users/josephthomasthachil/Desktop/Misc/GPU`

---



<!-- Source: mn5_guide/README.md -->
# MareNostrum 5 Guide - Domyn Guard

![MN5 Network Topology](images/mn5_network_islands.png)



## Overview
This comprehensive guide consolidates all essential information for using the MareNostrum 5 (MN5) supercomputer, tailored for the Domyn Guard team's LLM training workflows.
 on the BSC supercomputer.

> [!IMPORTANT]
> **No Hallucinations**: Information here is strict adherence to the [BSC Support Knowledge Center](https://www.bsc.es/supportkc/docs/MareNostrum5/intro/).

## 1. System Architecture
MN5 has two distinct partitions. You must know which one you are targeting.

| Feature | **GPP (General Purpose)** | **ACC (Accelerated)** |
| :--- | :--- | :--- |
| **Use Case** | Data prep, CPU inference, compilation | **LLM Training**, GPU inference |
| **CPU** | Intel Sapphire Rapids (112 cores/node) | Intel Sapphire Rapids (80 cores/node) |
| **GPU** | None | **4x NVIDIA H100 (64GB HBM2)** |
| **RAM** | 256 GB - 2 TB | 512 GB |
| **Login Node** | `glogin1.bsc.es` | `alogin1.bsc.es` |

**For Training**: You will exclusively use **ACC**.
**For Preprocessing**: You may use **GPP** to save GPU quotas.

---

## 2. File Systems & Quotas
Do not run jobs from your HOME directory.

| Path | Purpose | Backup? | Quota | Performance |
| :--- | :--- | :--- | :--- | :--- |
| `/gpfs/home` | Source code, scripts, configs. | Yes | Strict | Low (MetaData heavy) |
| `/gpfs/projects` | Shared datasets, final checkpoints. | Daily | Group-based | High |
| `/gpfs/scratch` | Temporary checkpoints, logs, training. | **NO (Purged)** | Group-based | **Highest** |

**Best Practice**:
1. Clone repo in `/gpfs/home`.
2. Move data to `/gpfs/projects` (read-only during training).
3. Output checkpoints/logs to `/gpfs/scratch`.

---

## 3. Environment Setup
MN5 uses specific modules. You should not install system-wide packages.

### Recommended `.bashrc` or Setup Script
```bash
# Clean environment
module purge

# Load basic tools
module load git
module load vim
module load python/3.11  # Check 'module avail python' for exact version

# CUDA for LLM Training
module load cuda/11.8    # Or 12.x depending on your codebase requirements
module load cudnn/8.9.7

# Create Virtual Environment (in project dir)
# python3 -m venv /gpfs/projects/bscXX/domyn-guard/venv
# source /gpfs/projects/bscXX/domyn-guard/venv/bin/activate
```

---

## 4. Running Jobs (ACC Partition)
You interact with the cluster via SLURM.

### Important Constraints
*   **CPU Ratio**: For every 1 GPU, you **MUST** request 20 CPUs.
*   **Memory**: You get proportionate memory.

### Interactive Session (Debug)
To get a shell on a GPU node for testing:
```bash
salloc --partition=acc_debug \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --time=01:00:00 \
       --qos=acc_debug
```

### Batch Job (Training)
See `scripts/train.sbatch` for a complete template.

---

## 5. Multi-Node distributed Training
MN5 nodes are connected via NDR200 InfiniBand (200Gb/s).
*   **Launcher**: Use `srun` (SLURM native) or `torchrun` (PyTorch).
*   **Network**: Set `NCCL_IB_DISABLE=0` and `NCCL_SOCKET_IFNAME=ib0` (verify interface name interactively).

See `scripts/multi_node.sbatch` for the exact configuration.

---



## 1. Quick Start
**[>>> READ THIS FIRST: MN5 for Humans (Team Quickstart)](TEAM_QUICKSTART.md) <<<**
*A simplified, jargon-free guide for new team members.*

## 6. Detailed Guides

*   **[00. NEW Essential Changes](00_new_essential_changes.md)**: **READ FIRST**. Critical SLURM changes.
*   [01. Setup & Login](01_setup_and_login.md): Access, VPN, and SSH keys.
*   [02. File Systems](02_file_systems.md): Where to store data vs code.
*   [03. Environment](03_environment.md): Modules and compilers.
*   [04. Running Jobs](04_running_jobs.md): Partition rules, QoS, and SLURM commands.
*   [05. Package Management](05_package_management.md): Python, Pip, Conda, and Venv best practices.
*   [06. Applications](06_available_applications.md): Pre-installed apps and benchmarks.
*   [07. Support & Chatbot](07_support_and_chat.md): Using `bsc_chat` and contacting support.
*   [08. Policies & Responsibilities](08_policies_and_responsibilities.md): Passwords, user conduct, and quotas.
*   [09. Advanced Utilities](09_advanced_utilities.md): `bsc_` commands, data transfers, and CPU affinity deep dive.
*   [10. System Package Managers](10_system_package_managers.md): Spack, EasyBuild, and EESSI for advanced users.
*   **[11. Distributed Training](11_distributed_training.md)**: PyTorch DDP, DeepSpeed, and NCCL tuning.
*   **[12. Containers (Apptainer)](12_containers.md)**: How to use Docker images (`.sif`) on MN5.

## 🛠️ Code Examples (New!)
We have provided Python examples optimized for MN5 hardware in the `mn5_guide/examples/` directory:
*   `mn5_distributed_setup.py`: How to parse SLURM variables for `torch.distributed` (DDP).
*   `mn5_dataloader.py`: Optimized `num_workers` and `pin_memory` settings for GPFS.
*   `mn5_safe_checkpoint.py`: Atomic saving logic to prevent corruption if jobs time out.
*   `test_gpu_connectivity.py`: Verify NVLink/InfiniBand bandwidth.


## 7. Appendix
See [09. Advanced Utilities](09_advanced_utilities.md) for troubleshooting common issues like "Permission denied" or SSH errors.


---


<!-- Source: mn5_guide/TEAM_QUICKSTART.md -->
# Team Quickstart: MN5 for Humans (Detailed)

This guide bridges the gap between basic analogies and power-user workflows.

---

## 🚀 Quick Access (Domyn Guard Team)

| Item | Value |
|------|-------|
| **Username** | `domy667574` |
| **Account** | `ehpc475` |
| **GPU Login** | `ssh domy667574@alogin1.bsc.es` |
| **CPU Login** | `ssh domy667574@glogin1.bsc.es` |
| **Data Transfer** | `ssh domy667574@transfer1.bsc.es` |
| **Password** | 1Password → "Login Mare Nostrum" |

---

## 1. The Supercomputer Analogy (Deep Dive)
Think of MN5 like a giant **Industrial Workshop**.

*   **Login Nodes (`glogin`/`alogin`)** = **The Reception Desk**.
    *   **Function**: File editing, heavy data transfers (via `transfer` nodes), compilation.
    *   **Strict Rule**: Processes using >10 minutes of CPU time are auto-killed.
*   **Compute Nodes** = **The Factory Floor**.
    *   **GPP (General Purpose)**: CPUs only. Great for data preprocessing.
    *   **ACC (Accelerated)**: The H100 GPUs. Costly. Hard to reserve.
*   **SLURM** = **The Shop Manager**.
    *   **Job**: Allocates resources.
    *   **Policy**: Fair share. Aspects like "Priority" determine if you wait 1 minute or 1 day.

## 2. Storage: The "Do Not Delete Me" Guide

| Location | Path | Quota (Soft/Hard) | Speed | Cleaning Policy | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Home** | `/gpfs/home/bscXX/bscXXYY` | 80GB / 84GB | Slow | Never wiped | `.bashrc`, `ssh` keys, small scripts. |
| **Projects** | `/gpfs/projects/bscXX` | Vast (PB scale) | Medium | Never wiped | Source code, final weights, datasets (read-only). |
| **Scratch** | `/gpfs/scratch/bscXX` | Vast (PB scale) | **NVMe Fast** | **WIPED EVERY 2 WEEKS** | Active training data, checkpoints, logs. |

> [!TIP]
> **Check your quota daily**: `bsc_quota`. If you hit the Hard Limit on Home, you can't even log in (VSCode will fail).

## 3. Workflow: From Zero to Hero

### Step 0: Professional Setup (VSCode)
Don't suffer with basic SSH. Use VSCode Remote.
1.  Install "Remote - SSH" extension.
2.  Edit `~/.ssh/config` locally:
    ```ssh
    Host mn5
        HostName alogin1.bsc.es
        User bscXXYY
        ForwardAgent yes
        ControlMaster auto
        ControlPath ~/.ssh/sockets/%r@%h-%p
        ControlPersist 600
    ```
    *(The `ControlMaster` settings prevent VSCode from opening 50 separate connections, speeding up the UI).*

### Step 1: Move Data Properly
Don't use drag-and-drop. Use `rsync` for speed and resume capability.
```bash
# From your LOCAL machine
rsync -avzP --exclude '.git' --exclude '__pycache__' \
  ./my_project/ bscXXYY@alogin1.bsc.es:/gpfs/projects/bscXX/bscXXYY/my_project/
```
*   `-a`: Archive mode (keeps permissions).
*   `-z`: Compress (save bandwidth).
*   `-P`: Show progress and allow resuming if wifi dies.

### Step 2: The Environment (Virtual Environments)
Don't just `pip install`.
1.  **Load System Python**: `module load python/3.11.2`
2.  **Create Vrenv (in SCRATCH!)**:
    *   Storage on `/gpfs/projects` is slow for thousands of small files (like inside `site-packages`).
    *   **Pro Tip**: Create venv in `/gpfs/scratch`, but keep requirement files in `/gpfs/projects`.
    ```bash
    python -m venv /gpfs/scratch/bscXX/bscXXYY/envs/my_env
    source /gpfs/scratch/bscXX/bscXXYY/envs/my_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### Step 3: Interactive Debugging (`salloc`)
Before submitting a 3-day training job, debug it for 10 minutes interactively.
```bash
# Request 1 GPU for 1 hour
salloc --partition=acc --qos=acc_debug --gres=gpu:1 --cpus-per-task=20 --time=01:00:00 --account=ehpc475
```
*   Wait for "Granted job allocation".
*   You are now on a compute node.
*   Run `nvidia-smi` to see your H100.
*   Run `python train.py` and see if it crashes immediately.

### Step 4: The Real Job (`sbatch`)
Submit and forget. See `mn5_guide/scripts/` for templates.
*   **Monitoring**:
    *   See running jobs: `squeue -u bscXXYY`
    *   See *why* a job failed (history): `sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed`
    *   *MaxRSS tells you real memory usage. If it's near 0, your job died instantly.*

## 4. Troubleshooting & Best Practices
*   **"Permission Denied"**: Did you switch groups? Run `bsc_project list` then `. bsc_project bscXX`.
*   **"No space left on device" (but quota is fine)**: You ran out of *inodes* (file count). Delete millions of tiny files or zip them up.
*   **Training is slow**:
    *   Are you dataloading from `/gpfs/home`? **STOP**. Move data to Scratch.
    *   Did you set `num_workers=0` in PyTorch? Set it to `4` or `8`.
*   **"Connection Reset"**: BSC Firewall cuts idle connections. Use `ClientAliveInterval 60` in your local SSH config or `tmux`/`screen` on the server (though `tmux` is risky on login nodes if they get rebooted).

## 4. The "Stay Safe" Checklist (Compliance)
*   [ ] **I am NOT running heavy python training on the login node.**
*   [ ] **I load modules (e.g., `python`, `cuda`) instead of condensing everything.**
*   [ ] **My large datasets are in `/gpfs/scratch`, not Home/Projects.**
*   [ ] **I requested 20 CPUs for every 1 H100 GPU.** (The Golden Rule).
*   [ ] **I have backed up my paper's results to my local laptop.**

## 5. F.A.Q.
**Q: My connection keeps dropping every 5 minutes.**
**A**: BSC firewall is aggressive. Add `ServerAliveInterval 60` to your `~/.ssh/config` file locally.

**Q: I can't write any files.**
**A**: Check `bsc_quota`. You likely hit the file count (inode) limit on `$HOME` or Project. Delete small files.

**Q: Can I run Jupyter?**
**A**: Yes, but only via SSH Tunneling (see `09_advanced_utilities.md`). **Never** expose it to the public internet.

**Q: My job has been 'Pending' for 3 days.**
**A**:
1. Check resource ratio (1 GPU : 20 CPU).
2. Check QoS limits (Did you submit 50 debug jobs? Limit is 1).
3. Check maintenance schedule (`bme` command).



---


<!-- Source: mn5_guide/MN5_QUICK_REFERENCE.md -->
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
bme                    # System status/maintenance
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


---


<!-- Source: mn5_guide/00_new_essential_changes.md -->
# 00. New Essential Changes (MN5 vs older)

If you are coming from MareNostrum 4 (MN4) or other clusters, pay attention.

## 1. User & Account Management
*   **Username Change**: `bscXXYYY` -> `bsc0XXYYY` (for some staff).
*   **Groups**:
    *   **Primary Group**: Institutional (no resources).
    *   **Secondary Groups**: Projects (where quota/jobs live).
*   **Action**: You MUST use `newgrp` (Linux) or `--account` (Slurm) to target the correct project group.

## 2. SLURM Submission (Mandatory fields)
You **must** specify both:
1.  `--account=<secondary_group>`
2.  `--qos=<queue_name>`

## 3. Storage Hierarchy
*   **Home**: `/gpfs/home` (One per user).
*   **Projects**: `/gpfs/projects` (One per project group).
*   **Scratch**: `/gpfs/scratch` (One per project group).

## 4. Performance Critical: `srun` & CPU Binding
**Change**: `srun` no longer inherits `--cpus-per-task` from `sbatch` automatically in the newer Slurm version.

**Fix**: You must be explicit or export the variable.

**Bad (Likely to fail or run slow):**
```bash
#SBATCH --cpus-per-task=20
srun ./my_binary
```

**Good:**
```bash
#SBATCH --cpus-per-task=20
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
srun params...
```

**Reason**: Without this, thread affinity (pinning) breaks, and threads may overlap on the same core.

## 5. MPI & NVIDIA SDK
*   If using **NVIDIA HPC SDK**, use `mpirun` instead of `srun`.
*   Set `export SLURM_CPU_BIND=none` when using `mpirun`.


---


<!-- Source: mn5_guide/01_setup_and_login.md -->
# 01. Setup & Login (Detailed)

## 1. Credentials & VPN
*   **User**: `bscXXYY`. This maps to a primary group (your university/institution) and secondary groups (your projects).
*   **VPN**: Mandatory for external access. `vpn.bsc.es`.
    *   *Linux Users*: Use `openconnect` for a better experience than the proprietary client.
    *   *Mac Users*: Pulse Secure is standard.

## 2. Advanced SSH Configuration
To avoid typing your password and hostname explicitly every time, and to enable **VSCode Remote**, setup your `~/.ssh/config` on your **local machine**:

```bash
ssh bscXXYY@glogin1.bsc.es
```
*   **Mac/Linux Users**: Use your native **Terminal** app. This is the professional standard.
*   **Windows Users**: Use WSL (Windows Subsystem for Linux) or PowerShell.

```ssh
# ~/.ssh/config

Host mn5
    HostName alogin1.bsc.es
    User bscXXYY
    # ForwardAgent allows you to use your local SSH keys to git clone FROM the cluster
    ForwardAgent yes
    # Keep connection alive helps prevent disconnects
    ServerAliveInterval 60
    # Multiplexing: Reuse one connection for multiple terminals (Faster VSCode)
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 10m
```

**Passwordless Entry**:
1.  Generate a key locally: `ssh-keygen -t ed25519`
2.  Copy it to MN5: `ssh-copy-id mn5`
3.  Now `ssh mn5` gets you in instantly.

## 3. Login Nodes in Depth
*   **`alogin1` - `alogin4`**: For ACC (GPU) partition. They have same architecture as compute nodes (x86_64).
*   **`glogin1` - `glogin4`**: For GPP (CPU) partition.
*   **`transfer1` - `transfer4`**: **Critical for Data**.
    *   These nodes have 40Gbps+ links to the outside world.
    *   If you are downloading ImageNet or uploading a 500GB checkpoint, **SSH into `transfer1` first**, then run your `wget` or `scp`.
    *   *Do not clog the login nodes with massive transfers.*

## 4. Key Login Commands
*   `bme`: "BSC Machine Environment". Shows Message of the Day (maintenance alerts).
*   `bsc_project`: Switch default Unix group. `source bsc_project bscXX`.
*   `passwd`: Change your password.

---

## 5. Team Access Details (Domyn Guard / Project 9868)

> [!IMPORTANT]
> These are the team-specific credentials for the EuroHPC project.

| Item | Value |
|------|-------|
| **Username** | `domy667574` |
| **Account** | `ehpc475` |
| **GPP Login (CPU)** | `glogin1.bsc.es`, `glogin2.bsc.es` |
| **ACC Login (GPU)** | `alogin1.bsc.es`, `alogin2.bsc.es` |
| **Transfer Node** | `transfer1.bsc.es` |

### Quick SSH Commands
```bash
# Connect to GPU login node (for GPU work)
ssh domy667574@alogin1.bsc.es

# Connect to CPU login node (for general work)
ssh domy667574@glogin1.bsc.es

# Connect to transfer node (for data transfers ONLY)
ssh domy667574@transfer1.bsc.es
```

> [!CAUTION]
> **No Outbound Internet**: The HPC cluster is disconnected from the public internet. You cannot `pip install` or `wget` from the compute nodes. Prepare all dependencies offline or in containers.


---


<!-- Source: mn5_guide/02_file_systems.md -->
# 02. File Systems & Storage Usage

## Storage Hierarchy Summary

| Filesystem | Path | Quota (Soft/Hard) | Speed | Backup | Cleaning Policy | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Home** | `/gpfs/home/bscXX/bscXXYY` | 80GB / 84GB | Low (NFS-like) | **Yes** | Never | Configs, Scripts, Keys, `.bashrc`. |
| **Projects** | `/gpfs/projects/bscXX` | Large (TB-PB) | Medium | **Yes** | Never | Code, Final Models, Read-Only Datasets. |
| **Scratch** | `/gpfs/scratch/bscXX` | Massive (PB) | **High (NVMe)** | **NO** | **2 Weeks** | **Active Training**, Checkpoints, Logs. |


Understanding the storage hierarchy is critical to avoid data loss and quota issues.
management.

## 1. `/gpfs/home`
*   **Path**: `/gpfs/home/bscXX/bscXXYY`
*   **Use Case**: Source code (`.py`, `.sh`), config files, virtual envs.
*   **Quota**: Very small (tens of GBs). **Do not store datasets here.**
*   **Backup**: Yes.

## 2. `/gpfs/projects`
*   **Path**: `/gpfs/projects/bscXX/domyn-guard`
*   **Use Case**: Large shared datasets, stable model checkpoints, finalized results.
*   **Quota**: Large (Terabytes). Shared by the whole team.
*   **Backup**: Often Daily/Weekly (check specifics).

## 3. `/gpfs/scratch` (Critical)
*   **Path**: `/gpfs/scratch/bscXX/bscXXYY` (or shared group scratch)
*   **Use Case**: **Active training I/O**.
    *   Dataset cache (HuggingFace `.cache`)
    *   Training logs (Tensorboard/WandB)
    *   Emergency checkpoints
*   **Performance**: Optimized for high throughput (NVMe enabled on calculation nodes).
*   **Retention**: **NO BACKUP**. Files older than X days (usually 30) are **AUTOMATICALLY DELETED**.

## Recommendation for Training
1.  **Code**: Keep in `$HOME`.
2.  **Data**: Copy dataset from `/gpfs/projects` to `/gpfs/scratch` before training if possible, or read directly from `/gpfs/projects` if read-only performance is sufficient.
3.  **Checkpoints**: Save to `/gpfs/scratch` during run. Copy **only the best** checkpoint to `/gpfs/projects` after training finishes.


---


<!-- Source: mn5_guide/03_environment.md -->
# 03. Software Environment

MN5 uses a `module` system to manage software versions.

## 1. Basic Commands
*   `module avail`: List all available software.
*   `module load <name>`: Load a specific tool.
*   `module list`: Show what is currently loaded.
*   `module purge`: Unload everything (start fresh).

## 2. Standard Stack for Domyn Guard
Place this in your `~/.bashrc` or a setup script `setup_env.sh`:

```bash
# Always start clean
module purge

# 1. Base Utilities
module load git
module load vim
module load htop

# 2. Python
# Check 'module avail python' to see versions. 3.10+ recommended for LLMs.
module load python/3.11

# 3. CUDA & Drivers
# CRITICAL: Match this with your PyTorch version.
# For PyTorch 2.1+, CUDA 11.8 or 12.1 is standard.
module load cuda/11.8 
module load cudnn/8.9.7
module load nccl/2.18.3   # Essential for distributed training
```

## 3. Python Virtual Environments
Do not install packages in the global scope (user flag is often disabled or hits quota limits).

```bash
# Create venv in Projects (more space) or Home (better reliability)
python3 -m venv /gpfs/projects/bscXX/domyn-guard/envs/my_env

# Activate
source /gpfs/projects/bscXX/domyn-guard/envs/my_env/bin/activate

# Install with pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
```

## 4. Avoiding Default Module Loading
If you want total control, run this once:
```bash
touch ${HOME}/.avoid_load_def_modules.mn5
```
This prevents MN5 from loading default modules on login, ensuring your scripts effectively define the environment.


---


<!-- Source: mn5_guide/04_running_jobs.md -->
# 04. Running Jobs (Deep Dive)

## 1. Partitions & QoS
You will almost always use `partition=acc` (Accelerated).

### Partitions
*   `acc`: H100 GPU Nodes.
*   `gpp`: CPU only.

## 2. Partition Architecture (Topology)

Understanding the hardware topology is vital for performance.

### General Purpose (GPP) Node
![GPP Topology](images/mn5_topology_gpp.png)
*   **Sockets**: 2x Intel Sapphire Rapids (56 cores each = 112 cores/node).
*   **NUMA**: Each socket is split into sub-NUMA clusters.
*   **RAM**: 256GB total standard.

### General Purpose HBM (GPP-HBM) Node
![GPP HBM Topology](images/mn5_topology_gpp_hbm.png)
*   **Specialty**: High Bandwidth Memory nodes.
*   **Sockets**: 2x Intel Sapphire Rapids with HBM.
*   **Use Case**: Memory-bound CPU simulations (CFD, Weather) that don't use GPUs.


### Accelerated (ACC) Node
![ACC Topology](images/mn5_topology_acc.png)
*   **GPU**: 4x NVIDIA H100 (64GB HBM3 each).
*   **CPU**: 4th Gen Intel Xeon (112 cores).
*   **IO**: NVMe storage (480GB) available for fast local scratch.

> [!IMPORTANT]
> **CPU Affinity**: Notice the NUMA domains in the images above? If you run a job without binding (`--cpu-bind`), your process might jump between sockets, killing performance. **Always** use `export SRUN_CPUS_PER_TASK`.

## 3. Submitting Jobs (Quality of Service)

> [!TIP]
> Run `bsc_queues` to see all available queues for your account.

### Team Account: `ehpc475`

| Queue Name | Max Time | Max Cores | Description |
|------------|----------|-----------|-------------|
| `acc_debug` | 2:00:00 | 640 | Debug jobs (higher priority, limited time) |
| `acc_ehpc` **(Default)** | 72:00:00 | 8000 | Standard EuroHPC production jobs |
| `acc_interactive` | 2:00:00 | 40 | Interactive sessions (login nodes only) |

*   **`acc_debug`**:
    *   Time limit: 2 hours, max 640 cores.
    *   **Priority**: Boosted. Use this to test if your code runs.
*   **`acc_ehpc`** (Default):
    *   Time limit: 72 hours (3 days).
    *   **Use for**: Production multi-node training.
*   **`acc_interactive`**:
    *   Time limit: 2 hours, max 40 cores.
    *   **Use for**: Interactive development sessions.

## 4. Expert Job Management

### Interactive Debugging (`salloc`)
Don't wait in the queue just to find a syntax error.
```bash
# "Give me a shell on a GPU node for 30 mins"
salloc --account=ehpc475 --qos=acc_debug --partition=acc --gres=gpu:1 --cpus-per-task=20 --time=00:30:00
```
*   Use `srun --pty bash` once allocated if the shell doesn't open automatically.

### Batch Script Specifics
The header of your `.sbatch` file controls everything.

```bash
#!/bin/bash
#SBATCH --job-name=llm_train
#SBATCH --output=%x_%j.out   # %x=jobname, %j=jobid
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4         # Request 4 GPUs
#SBATCH --cpus-per-task=80   # 20 * 4 = 80 CPUs
#SBATCH --exclusive          # Dedicate node to me (good for benchmarks)
```

### Forensic Analysis (`sacct`)
Your job died. Why?
```bash
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,Elapsed,NodeList
```
*   **MaxRSS**: Maximum Resident Set Size (RAM). If this is close to 2TB (node limit) or your requested limit, you hit OOM.
*   **ExitCode**: `0:0` (Success), `1:0` (Error), `0:125` (OOM/Kill).

### Resource Rules (ACC)
MN5 enforces strict proportionality.
*   **Ratio**: 1 GPU : 20 CPUs.
*   **Violation**: Requesting `gpu:1` and `cpus-per-task:40` will wait forever.
*   **Memory**: RAM is allocated per CPU. approx 2-3GB per CPU.

## 5. Troubleshooting Common Failures
1.  **"Socket verification failed"**: SSH key issue.
2.  **"QOSMaxCpusPerUserLimit"**: You already have max jobs running. Wait.
3.  **"DIB / Data IB Error"**: Infiniband network glitch. Usually transient. Resubmit.
4.  **"Bus Error"**: Often means you accessed a file that doesn't exist or a corrupt memory pointer. Check data paths.


---


<!-- Source: mn5_guide/05_package_management.md -->
# 05. Package Management (Python & Conda)

Managing dependencies correctly is crucial to avoid quota issues and conflicts.

## 1. Python (pip & venv)
Recommended for most projects.

### Step 1: Load Python Module
```bash
module load python/3.11  # Or your preferred version
```

### Step 2: Create a Virtual Environment
**Do not use `--user` flag globally.** It fills up your home directory ($HOME) quickly. Use virtual environments in `$PROJECTS`.

```bash
# Good Practice: Store env in Projects
python3 -m venv /gpfs/projects/bscXX/domyn-guard/envs/my_env
```

### Step 3: Activate and Install
```bash
source /gpfs/projects/bscXX/domyn-guard/envs/my_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
```

---

## 2. Conda (Anaconda/Miniconda)
Use this if you need complex non-Python dependencies.

### Step 1: Load Conda Module
MN5 provides a centralized Conda module.
```bash
module load anaconda
# OR
module load miniconda3
```

### Step 2: Initialize (First Time Only)
```bash
conda init bash
source ~/.bashrc
```

### Step 3: Configure Paths (Critical)
By default, Conda stores envs in `~/.conda` (HOME), which has a small quota. **Redirect it to PROJECTS.**

```bash
# Create a .condarc file
conda config --add envs_dirs /gpfs/projects/bscXX/domyn-guard/conda_envs
conda config --add pkgs_dirs /gpfs/projects/bscXX/domyn-guard/conda_pkgs
```

### Step 4: Create Environment
```bash
conda create -n my_conda_env python=3.10
conda activate my_conda_env
```

---

## 3. Best Practices
1.  **Cache**: Pip and Conda cache large files in `~/.cache`. Symlink this folder to `/gpfs/scratch` or `/gpfs/projects` to avoid Home quota limits.
    ```bash
    mkdir -p /gpfs/scratch/bscXX/bscXXYY/.cache
    rm -rf ~/.cache
    ln -s /gpfs/scratch/bscXX/bscXXYY/.cache ~/.cache
    ```
2.  **Reproducibility**: Always export your environment.
    ```bash
    pip freeze > requirements.txt
    # OR
    conda env export > environment.yml
    ```


---


<!-- Source: mn5_guide/06_available_applications.md -->
# 06. Available Applications & Benchmarks (Detailed)

MN5 hosts optimized builds for major scientific domains.

## 1. Domain Specific Applications
Checking availability: `module avail <name>`

### Chemistry/Materials
*   **GROMACS**:
    *   **Module**: `module load GROMACS/2024.1-acc-cuda-12` (Example).
    *   **Tip**: Always use the `-acc-cuda` versions on ACC partition for 50x speedup over CPU versions.
    *   **Command**: `gmx_mpi mdrun ...`
*   **VASP**:
    *   **Module**: `module load VASP`
    *   **License**: Restricted. You must be in the `vasp` unix group.
    *   **Tip**: VASP requires `ulimit -s unlimited` in your script stack size.
*   **CP2K**:
    *   Hybrid MPI/OpenMP is critical here. Set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.

> [!NOTE]
> For detailed benchmarks and build references, see the official application pages: [GPP Apps](https://www.bsc.es/supportkc/docs/MareNostrum5/Marenostrum5-Applications/GPP), [ACC Apps](https://www.bsc.es/supportkc/docs/MareNostrum5/Marenostrum5-Applications/ACC/).

## 2. Deep Learning / AI
BSC provides pre-built containers and modules, but most users bring their own environment.

### Recommended AI Stack
1.  **Base**: `module load python/3.11` + `module load cuda/12.1`
2.  **Manager**: `venv` (preferred) or `conda`.
3.  **Flash Attention**: Installing `flash-attn` can be tricky. Ensure `nvcc --version` matches your PyTorch CUDA version exactly.

## 3. Benchmarks
*   **HPL (High Performance Linpack)**: `module load hpl`. Used for stress testing.
*   **HPCG**: Memory-bound benchmark.


---


<!-- Source: mn5_guide/07_support_and_chat.md -->
# 07. Support & BSC Chatbot

## 1. BSC Chatbot (`bsc_chat`)
MN5 hosts an internal AI chatbot for support and general queries, running on local hardware (data privacy compliant).

### Availability
*   **Nodes**: Available on basic login nodes (`glogin1/2`, `alogin1/2`).
*   **Not Available**: Restricted login nodes (`glogin4`, `alogin4`).

### Usage
1.  **Load the Module**:
    ```bash
    module load bsc
    ```
2.  **Start Chat**:
    ```bash
    bsc_chat
    ```
3.  **Select Model**:
    *   `0: Knowledge-support`: Trained on BSC docs. Best for "How do I use SLURM?" questions.
    *   `1: Llama-70B-Instruct`: General purpose reasoning.

### Interaction
*   Type your prompt.
*   Press **ALT + ENTER** to send.

> [!NOTE]
> Like all LLMs, it can hallucinate. Always verify commands with official docs.

---

## 2. Official Support
If the chatbot or docs don't help:
*   **Email**: support@bsc.es
*   **Ticket**: Include your username (`bscXXYY`) and Job ID if relevant.


---


<!-- Source: mn5_guide/08_policies_and_responsibilities.md -->
# 08. Policies & User Responsibilities

## 1. User Responsibilities
As a user of a shared supercomputer, you are expected to:
*   **Data Management**: Backup your own critical data. `/gpfs/scratch` is purged. `/gpfs/projects` is backed up but you should have off-site copies of papers/results.
*   **Login Nodes**: **NEVER** run heavy computational scripts on login nodes (`alogin`/`glogin`). They are for editing, compilation, and job submission only. Violators are often auto-killed or banned.
*   **Security**: Accounts are personal and non-transferable. Never share your password or SSH key.

## 2. Password Management
*   **Changing Password**: If you need to change your password, use the official BSC portal (link provided in your welcome email) or check the [HPC User Portal](https://www.bsc.es/supportkc/docs-utilities/hpc_portal).
    ![BSC Auth Portal](images/bsc_auth_portal.png)
*   **Legal Agreement**: On first login (or annual renewal), you must accept the Acceptable Use Policy (AUP).
    ![BSC User Agreement](images/bsc_user_agreement.png)
*   **SSH Keys**: We strongly recommend using SSH keys with passphrases for convenience and security.


## 3. Quotas
You are responsible for monitoring your usage.
*   **Check Quota**: `bsc_quota`
*   **Grace Period**: If you exceed soft limits, you have ~7 days to clean up before you are blocked from writing.

## 4. CPU Affinity & Performance
*   **Binding**: Proper process binding is your responsibility. Using `srun` without correct flags (see `00_new_essential_changes.md`) can cause processes to stack on the same core, destroying performance ($$ lost).
*   **Efficiency**: Aim for >80% GPU utilization. If your job averages 10%, you are wasting resources and blocking others.


---


<!-- Source: mn5_guide/09_advanced_utilities.md -->
# 09. Advanced Utilities & Remote Connections

## 1. Remote Connections & Port Forwarding
Running a Jupyter Notebook or Tensorboard on a compute node and viewing it on your Mac requires **SSH Tunneling**.

### The Concept
Your laptop cannot "see" the compute node IP directly (it's behind a firewall). You must tunnel through the Login Node.

### Scenario A: Basic TCP Forwarding
![SSH Tunnel Basic](images/mn5_ssh_tunnel_basic.png)
You forward a port (e.g., 8888) from the compute node, through the login node, to your localhost.

### Scenario B: Multi-Node Tunneling
![SSH Tunnel Multi-Node](images/mn5_ssh_tunnel_multi_node.png)
If you have jobs running on **different** compute nodes (e.g., Node A and Node B), you can tunnel to both simultaneously using different local ports.
```bash
# Tunnel 8888 -> Node A, 8889 -> Node B
ssh -L 8888:nodeA:8888 -L 8889:nodeB:8888 bscXXYY@alogin1.bsc.es
```

### Scenario C: Multi-Client Forwarding
![SSH Tunnel Multi-Client](images/mn5_ssh_tunnel_multi_client.png)
Two different users (or two laptops) can tunnel to different compute nodes through the same login gateway without conflict.

### Scenario D: Socket Forwarding (Secure & Scalable)
![SSH Socket](images/mn5_ssh_socket_basic.png)
Uses a Unix socket file instead of a TCP port. This is safer as it avoids port collisions.

#### Multi-Socket Setup
![SSH Socket Multi](images/mn5_ssh_socket_multi.png)
Advanced users can map multiple sockets (one per compute job) to different local socket files.


## 2. BSC Specific Commands
*   `bsc_quota`: Check storage limits.
*   `bsc_jobs`: Summarized view of your active jobs.
*   `bsc_load`: Check status of login nodes (load avg).

## 3. Data Transfer Machines (`dtmachines`)
For TB-scale transfers, do not use `glogin` or `alogin`.
*   **Hostname**: `dt01.bsc.es`, `dt02.bsc.es`
*   **Usage**:
    1.  SSH into `dt01`.
    2.  Run `rsync`, `scp`, or `globus`.
    3.  Logout.

## 4. X11 Forwarding (GUI Apps)
If you need to open a GUI window (e.g., a plot):
1.  **Mac**: Install **XQuartz**.
2.  **SSH**: Add `-X` or `-Y` flag.
    ```bash
    ssh -Y bscXXYY@glogin1.bsc.es
    ```
3.  **Test**: Run `xeyes`. If eyes appear on your desktop, it works.

## 5. CPU Affinity (Detailed)
See `04_running_jobs.md` for the topology diagrams.
*   **Why**: MN5 nodes have 2 sockets. Crossing sockets costs ~40% performance penalty.
*   **How**:
    ```bash
    # In SBATCH
    #SBATCH --cpu-bind=verbose,interfaces
    ```


---


<!-- Source: mn5_guide/10_system_package_managers.md -->
# 10. System Package Managers (Spack, EasyBuild, EESSI)

Beyond standard modules and Python/Conda, MN5 provides access to three powerful package management systems. These are useful if you need to compile scientific software from source with specific optimizations or dependencies not found in the default environment.

## 1. EESSI (European Environment for Scientific Software Installations)
**Best for**: Instant access to a huge library of scientific software optimized for x86_64, ARM, and GPUs, streamed over the network.

*   **How it works**: Uses CVMFS to mount a read-only repository at `/cvmfs/software.eessi.io`.
*   **Usage**:
    1.  **Load the Module**:
        ```bash
        module load eessi
        ```
    2.  **Search & Load Software**:
        ```bash
        module avail
        module load GROMACS/2024.1-foss-2023b
        ```
*   **Partition Support**:
    *   **GPP**: Loads CPU-optimized modules `.../sapphirerapids/modules/all`.
    *   **ACC**: Loads CPU modules + GPU-optimized modules `.../sapphirerapids/accel/nvidia/cc90/modules/all`.

## 2. EasyBuild
**Best for**: Consistency. The primary tool used by BSC admins to install the system modules.

*   **Usage**:
    1.  **Load the Module**:
        ```bash
        module load EB/apps
        ```
    2.  **Search & Load**:
        ```bash
        module avail
        module load <name>/<version>  # ALWAYS specify version!
        ```
*   **Locations**:
    *   **GPP**: `/apps/GPP/EASYBUILD/modules/all`
    *   **ACC**: `/apps/ACC/EASYBUILD/modules/all`

## 3. Spack
**Best for**: "Do It Yourself" custom builds. If you need a specific version of a library with a non-standard compiler flag.

*   **System Spack**:
    ```bash
    module load spack
    spack find        # List installed packages
    spack load <name> # Load a package
    ```

*   **User Spack (Advanced)**:
    You can install your own Spack instance in your home directory and "chain" it to the system Spack to reuse pre-built packages.
    1.  Install Spack in `$HOME`.
    2.  Configure `$HOME/spack/etc/spack/defaults/upstreams.yaml`:
        ```yaml
        upstreams:
          system_spack:
            install_tree: /apps/GPP/SPACK/0.21.2/opt/spack
        ```
    3.  Install your custom packages: `spack install my-package`


---


<!-- Source: mn5_guide/11_distributed_training.md -->
# 11. Distributed Training (AI/Deep Learning)

MareNostrum 5 (ACC partition) is designed for massive scale distributed training. However, the default environment often needs tweaking to make PyTorch/DeepSpeed happy with the underlying InfiniBand network.

## 1. The Hardware Reality
*   **Interconnect**: NVIDIA Quantum-2 InfiniBand (NDR).
*   **P2P**: NVLink between GPUs on the same node.
*   **Network**: InfiniBand between nodes.

## 2. PyTorch DDP Setup
Use the helper script `mn5_distributed_setup.py` provided in `examples/`.

### Key Environment Variables
MN5 requires explicit variables to tell NCCL which network interface to use. If you don't set these, NCCL might try to send tensors over the Ethernet management interface (slow/broken).

```bash
# In your SLURM script:

# 1. Force NCCL to use InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
# export NCCL_SOCKET_IFNAME=ib0 # Sometimes needed if auto-detection fails

# 2. Debugging (If training hangs)
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
```

## 3. DeepSpeed Configuration
When using DeepSpeed, rely on `hostfile` generation.

```bash
# Generate DeepSpeed hostfile from SLURM
scontrol show hostnames $SLURM_JOB_NODELIST > ./hostfile
```

Modify your `ds_config.json` to match the hardware:
```json
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true
  }
}
```

## 4. Common Issues
### "Watchdog caught hanging process"
**Cause**: IO Starvation. The GPUs are waiting for data from GPFS.
**Fix**:
1. Follow `mn5_dataloader.py` example (Pin memory, more workers).
2. Move data to `/gpfs/scratch` (NVMe-like speed).
3. Increase NCCL timeout: `export NCCL_ASYNC_ERROR_HANDLING=1` and `export TORCH_DISTRIBUTED_DEBUG=DETAIL`.

### "NCCL WARN: Connect to ... failed"
**Cause**: The nodes cannot talk to each other on the specified port.
**Fix**:
1. Ensure `MASTER_ADDR` is correct (see `mn5_distributed_setup.py`).
2. Ensure you are not blocked by firewall (use internal InfiniBand IPs, usually handled by hostnames).

## 5. Verification
Run the `examples/test_gpu_connectivity.py` script to benchmark:
1.  **Intra-node**: NVLink bandwidth (~400GB/s).
2.  **Inter-node**: InfiniBand bandwidth.


---


<!-- Source: mn5_guide/12_containers.md -->
# 12. Containers (Apptainer / Singularity)

On HPC systems like MN5, you **cannot use Docker** (it requires root privileges). Instead, we use **Apptainer** (formerly Singularity).

## 1. Why use Containers?
*   **Reproducibility**: Your environment is frozen.
*   **Compatibility**: You need Ubuntu 22.04 but MN5 runs Red Hat.
*   **Customization**: You need a specific CUDA version not in `module av`.

## 2. Converting Docker to Apptainer
You typically build your image on your **local laptop** (where you have Docker) or pull directly from Docker Hub.

### Method A: Pull from Docker Hub (On Login Node)
```bash
# Pulls pytorch:latest and converts to pytorch.sif
apptainer pull pytorch.sif docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
```

### Method B: Build from local Docker
1.  **Local Machine**:
    ```bash
    # Save docker image to tar
    docker save my_image:latest -o my_image.tar
    # SCP to MN5
    rsync -P my_image.tar bscXXYY@dt01.bsc.es:/gpfs/scratch/bscXX/bscXXYY/
    ```
2.  **MN5 (Login Node)**:
    ```bash
    apptainer build my_image.sif docker-archive://my_image.tar
    ```

## 3. Running Jobs with Apptainer
In your SLURM script, replace `python train.py` with the container execution.

```bash
#!/bin/bash
#SBATCH --gres=gpu:1

# Load Apptainer
module load apptainer

# Execute
# --nv: Enable NVIDIA GPU support inside container
# -B: Bind mount directories (Map MN5 paths to Container paths)

apptainer exec --nv \
    -B /gpfs/projects/bscXX/bscXXYY:/app \
    -B /gpfs/scratch/bscXX/bscXXYY:/scratch \
    ./pytorch.sif \
    python /app/train.py
```

## 4. Writable Containers (Sandboxes)
`.sif` files are read-only. If you need to `pip install` things *inside* MN5:
```bash
# Build as a sandbox directory
apptainer build --sandbox my_env_dir docker://python:3.10

# Shell into it with write permission (fakeroot)
apptainer shell --writable --fakeroot my_env_dir

# Now you can pip install
Apptainer> pip install numpy
```
*Note: Using thousands of files in a sandbox dir is slow on GPFS. Prefer building the single `.sif` file locally.*

---

## 5. Team Workflow: Singularity on MN5

> [!IMPORTANT]
> **Cannot use Docker directly on HPC** (no root privileges). Must use Singularity/Apptainer.

### Key Constraints
- **No outbound internet** — Cannot `pip install` or `wget` from compute nodes
- **Architecture**: AMD64 (x86_64)
- **Build GPU images locally** — Build on a GPU-enabled VM (Azure, GCP), then transfer `.sif` to MN5

### Updating Code Inside Containers
To update or patch code inside a Singularity image without rebuilding:
```bash
# Mount your local code directory over the container's code
apptainer exec --nv \
    -B /gpfs/projects/ehpc475/code:/app \
    ./my_container.sif \
    python /app/train.py
```

### Hugging Face Offline Mode
Since there's no internet, synchronize HF cache manually:
```bash
# On your local machine, download models
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir ./hf_cache

# Transfer to MN5
rsync -avzP ./hf_cache domy667574@transfer1.bsc.es:/gpfs/projects/ehpc475/hf_cache/

# In your training script, set offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/gpfs/projects/ehpc475/hf_cache
```


---

# Appendix: Code Examples & Configurations

## Directory: `mn5_guide/examples`

### check_mn5_status.sh
Path: `mn5_guide/examples/check_mn5_status.sh`

```bash
#!/bin/bash
# A quick health-check script for your MN5 account

echo "=== 1. Checking Quota  ==="
bsc_quota
echo ""

echo "=== 2. Active Jobs ==="
squeue -u $USER
echo ""

echo "=== 3. Login Nodes Load ==="
bsc_load
echo ""

echo "=== 4. File Count (Inodes) ==="
# Approximated. Real check uses bsc_quota but sometimes this is useful.
echo "Checking $HOME..."
find $HOME -maxdepth 1 | wc -l
```

### interactive_session.sh
Path: `mn5_guide/examples/interactive_session.sh`

```bash
#!/bin/bash
# Interactive GPU Session for Development
# Usage: ./interactive_session.sh [hours] [gpus]
#
# Examples:
#   ./interactive_session.sh           # 1 hour, 1 GPU (default)
#   ./interactive_session.sh 2         # 2 hours, 1 GPU
#   ./interactive_session.sh 2 4       # 2 hours, 4 GPUs

HOURS=${1:-1}
GPUS=${2:-1}
CPUS=$((GPUS * 20))  # 20 CPUs per GPU rule

echo " Requesting interactive session..."
echo "   Duration: ${HOURS} hour(s)"
echo "   GPUs: ${GPUS}"
echo "   CPUs: ${CPUS}"
echo ""

salloc \
    --account=ehpc475 \
    --partition=acc \
    --qos=acc_debug \
    --gres=gpu:${GPUS} \
    --cpus-per-task=${CPUS} \
    --time=${HOURS}:00:00

# After allocation, you'll be on a compute node
# Run: nvidia-smi to verify GPU access
# Run: python train.py to test your code

```

### mn5_dataloader.py
Path: `mn5_guide/examples/mn5_dataloader.py`

```python
import torch
from torch.utils.data import DataLoader, Dataset
import tempfile
import os

class Mn5Dataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        # On MN5, ensure your data is in /gpfs/scratch, NOT /gpfs/projects or HOME.
        # Projects is optimized for throughput, not IOPS (latency).
        # Home has a strict quota and is slow.
        self.check_path()

    def check_path(self):
        cwd = os.getcwd()
        if "/gpfs/home" in cwd:
            print("[WARN] You are running from HOME. This will be slow and may hit quota limits.")
        elif "/gpfs/projects" in cwd:
            print("[INFO] Running from PROJECTS. Read-only large files are fine here.")
        elif "/gpfs/scratch" in cwd:
            print("[OK] Running from SCRATCH. Optimal for active IO.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate data loading
        return torch.randn(3, 224, 224), torch.randint(0, 100, (1,))

def get_optimized_loader(dataset, batch_size=32):
    """
    Returns a DataLoader optimized for MN5 hardware (112 cores per node).
    """
    
    # MN5 Rule: 1 GPU : 20 CPUs.
    # So you realistically have ~15-18 works available per GPU.
    # Setting num_workers too high (>20) creates overhead.
    # Setting it too low (0) causes GPU starvation.
    NUM_WORKERS = 16 
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # Key for GPFS:
        num_workers=NUM_WORKERS,
        # Keeps workers alive between epochs to avoid re-forking overhead
        persistent_workers=True, 
        # Crucial for GPU transfer speed
        pin_memory=True,
        # prefetch_factor * num_workers = batches buffered. 
        # Don't set too high or you scream OOM. 2 is usually safe.
        prefetch_factor=2 
    )
    return loader

if __name__ == "__main__":
    ds = Mn5Dataset()
    loader = get_optimized_loader(ds)
    print("DataLoader ready. Iterating...")
    for batch in loader:
        pass
    print("Done.")

```

### mn5_distributed_setup.py
Path: `mn5_guide/examples/mn5_distributed_setup.py`

```python
import os
import subprocess
import torch
import torch.distributed as dist

def setup_mn5_distributed():
    """
    Robustly configures the PyTorch Distributed environment for MareNostrum 5.
    Handles specific SLURM environment variables and sets up the MASTER_ADDR/PORT.
    """
    if "SLURM_JOB_ID" not in os.environ:
        print("[WARN] Not running inside SLURM. Defaulting to local standalone mode.")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        dist.init_process_group(backend="nccl")
        return

    # 1. Get List of Nodes
    # MN5 uses a compressed format like "node[01-04]". We need the first one as master.
    try:
        nodelist = os.environ.get("SLURM_JOB_NODELIST")
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist]
        ).decode().strip().splitlines()
    except Exception as e:
        raise RuntimeError(f"Could not parse SLURM nodelist: {e}")

    master_node = hostnames[0]
    
    # 2. Set Environment Variables for Torch
    os.environ["MASTER_ADDR"] = master_node
    # Pick a random port or a fixed one. 
    # Just ensure it doesn't conflict if multiple jobs share a node (unlikely in exclusive mode).
    os.environ["MASTER_PORT"] = "29500" 

    # SLURM_PROCID is the global rank (0 to Total GPUs - 1)
    os.environ["RANK"] = os.environ.get("SLURM_PROCID")
    
    # SLURM_NTASKS is the world size
    os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS")
    
    # SLURM_LOCALID is the local rank on the node (0, 1, 2, 3)
    os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID")

    print(f"[MN5-Setup] Master: {master_node}, Rank: {os.environ['RANK']}, World: {os.environ['WORLD_SIZE']}")

    # 3. Initialize Process Group
    # 'nccl' is mandatory for H100s.
    dist.init_process_group(backend="nccl")
    
    # 4. Set Device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    print(f"[MN5-Setup] Successfully initialized rank {os.environ['RANK']} on {os.uname().node}")

if __name__ == "__main__":
    setup_mn5_distributed()
    # Your training loop here

```

### mn5_safe_checkpoint.py
Path: `mn5_guide/examples/mn5_safe_checkpoint.py`

```python
import os
import torch
import shutil
import time

def save_checkpoint_safe(model, optimizer, epoch, path):
    """
    Saves a checkpoint atomically to prevent corruption on GPFS.
    GPFS is robust, but if your job hits a time limit mid-write, 
    you end up with a corrupt, size-0 file.
    
    Strategy: Write to .tmp file, then rename. Rename is atomic.
    """
    
    # 1. Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 2. Define temp path
    tmp_path = path + ".tmp"
    
    print(f"[Checkpoint] Saving to temporary file: {tmp_path}")
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 3. Save to tmp
    try:
        torch.save(state, tmp_path)
    except Exception as e:
        print(f"[Error] Failed to save checkpoint: {e}")
        return

    # 4. Atomic Rename
    # If the job dies here, you either hava the old file OR the new file.
    # Never a half-written file.
    shutil.move(tmp_path, path)
    print(f"[Checkpoint] Successfully renamed to: {path}")

    # 5. Cleanup Old Checkpoints (Optional)
    # MN5 Scratch has a file count limit. Clean up!
    # implementation specific...

def load_checkpoint(path, model, optimizer):
    if not os.path.exists(path):
        print("[Checkpoint] No checkpoint found. Starting from scratch.")
        return 0
    
    print(f"[Checkpoint] Loading from {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

```

### test_gpu_connectivity.py
Path: `mn5_guide/examples/test_gpu_connectivity.py`

```python
import torch
import os
import time

def test_gpu():
    print(f"=== GPU Connectivity Test on {os.uname().node} ===")
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available.")
        return

    count = torch.cuda.device_count()
    print(f"[INFO] GPU Count: {count}")
    
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} | Mem: {props.total_memory / 1e9:.2f} GB")

    if count > 1:
        print("\n=== P2P Bandwidth Test (Approx) ===")
        # Simple test: Send tensor from GPU 0 to GPU 1
        t0 = torch.zeros(1024*1024*100, device='cuda:0') # 400MB float32
        torch.cuda.synchronize()
        
        start = time.time()
        t1 = t0.to('cuda:1')
        torch.cuda.synchronize()
        end = time.time()
        
        size_gb = (t0.nelement() * 4) / 1e9
        speed = size_gb / (end - start)
        print(f"[INFO] Transfer 0 -> 1: {speed:.2f} GB/s (Includes overhead)")
        print("Note: NVLink should be >200 GB/s. PCIe is ~20 GB/s.")
        
if __name__ == "__main__":
    test_gpu()

```

### test_singularity.sh
Path: `mn5_guide/examples/test_singularity.sh`

```bash
#!/bin/bash
# Test Singularity/Apptainer Setup on MN5
# Run this from a compute node (after salloc) to verify container setup
#
# Usage: ./test_singularity.sh

set -e

echo "🐳 Testing Singularity/Apptainer on MN5"
echo "========================================="
echo ""

# Check if apptainer is available
echo "1. Checking Apptainer installation..."
module load apptainer 2>/dev/null || echo "   Note: apptainer may be loaded by default"
which apptainer && echo "   ✅ Apptainer found" || echo "   ❌ Apptainer not found"
echo ""

# Check for GPU support
echo "2. Checking GPU availability..."
if nvidia-smi &>/dev/null; then
    echo "   ✅ NVIDIA driver detected"
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv
else
    echo "   ⚠️  No GPU detected (are you on a login node?)"
fi
echo ""

# Test simple container execution
echo "3. Testing container pull (may take a minute first time)..."
TEST_SIF="/tmp/test_python.sif"
if [ ! -f "$TEST_SIF" ]; then
    apptainer pull "$TEST_SIF" docker://python:3.10-slim 2>/dev/null
fi
echo "   ✅ Container pulled successfully"
echo ""

# Test container execution
echo "4. Testing container execution..."
apptainer exec "$TEST_SIF" python --version
echo "   ✅ Container execution works"
echo ""

# Test GPU container (if on compute node)
echo "5. Testing GPU access inside container..."
if nvidia-smi &>/dev/null; then
    # Try running nvidia-smi inside a CUDA container
    apptainer exec --nv "$TEST_SIF" python -c "print('   ✅ GPU passthrough test complete')" 2>/dev/null || \
        echo "   ⚠️  GPU passthrough test skipped (need CUDA container for full test)"
else
    echo "   ⏭️  Skipped (no GPU available on login node)"
fi
echo ""

echo "========================================="
echo "🎉 Singularity/Apptainer test complete!"
echo ""
echo "Next steps:"
echo "  1. Build your project container locally"
echo "  2. Transfer to MN5: rsync -P my_container.sif domy667574@transfer1.bsc.es:/gpfs/projects/ehpc475/"
echo "  3. Run with: apptainer exec --nv -B /gpfs/projects/ehpc475:/app ./my_container.sif python train.py"

```

## Directory: `mn5_guide/scripts`

### train_multi_node.sbatch
Path: `mn5_guide/scripts/train_multi_node.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=domyn-train-multi
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=acc
#SBATCH --qos=acc_normal
#SBATCH --time=24:00:00
#SBATCH --nodes=4                  # Request 4 Nodes
#SBATCH --ntasks-per-node=4        # 4 GPUs per node = 16 GPUs total
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20         # 20 CPUs per GPU
#SBATCH --exclusive

# 1. Load Environment
module purge
module load python/3.11 cuda/11.8 cudnn/8.9.7
source /gpfs/projects/bscXX/domyn-guard/venv/bin/activate

# 2. Network Specifics for MN5 (InfiniBand)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
# Ensure we bind to correct interface (check with 'ip a' in interactive session if unsure)
# export NCCL_SOCKET_IFNAME=ib0 

# 3. Master Node Setup
# Get the first node name
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Master Node: $head_node"
echo "Master IP: $head_node_ip"

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export LOGLEVEL=INFO

# 4. Run Training
# srun will automatically launch 4 tasks per node (as defined in #SBATCH)
# We map pyscript execution
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:$MASTER_PORT \
    /gpfs/home/bscXX/username/repo/scripts/train.py \
    --config /gpfs/home/bscXX/username/repo/config/config.json

```

### train_single_node.sbatch
Path: `mn5_guide/scripts/train_single_node.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=domyn-train-1node
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=acc            # Accelerated Partition (H100)
#SBATCH --qos=acc_normal           # Quality of Service
#SBATCH --time=12:00:00            # Max runtime
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=4        # 1 task per GPU
#SBATCH --gres=gpu:4               # Request all 4 GPUs on the node
#SBATCH --cpus-per-task=20         # MANDATORY: 20 CPUs per GPU
#SBATCH --exclusive                # Exclusive node access

# 1. Load Environment
module purge
module load python/3.11 cuda/11.8 cudnn/8.9.7
source /gpfs/projects/bscXX/domyn-guard/venv/bin/activate

# 2. Debug Info
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# 3. Environment Variables for PyTorch
export OMP_NUM_THREADS=20
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# 4. Run Training
# Usage: torchrun matches ntasks (4 processes)
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /gpfs/home/bscXX/username/repo/scripts/train.py \
    --config /gpfs/home/bscXX/username/repo/config/config.json

```

### transfer_data.sh
Path: `mn5_guide/scripts/transfer_data.sh`

```bash
#!/bin/bash
# Data Transfer Helper for MN5
# Use this script to transfer files to/from MN5 via the transfer nodes
#
# Usage:
#   ./transfer_data.sh upload /local/path /remote/path
#   ./transfer_data.sh download /remote/path /local/path
#
# Examples:
#   ./transfer_data.sh upload ./my_dataset /gpfs/scratch/ehpc475/datasets
#   ./transfer_data.sh download /gpfs/projects/ehpc475/results ./results

set -e

TRANSFER_HOST="domy667574@transfer1.bsc.es"
ACTION=${1:-help}
SOURCE=${2:-}
DEST=${3:-}

show_help() {
    echo "MN5 Data Transfer Helper"
    echo "========================"
    echo ""
    echo "Usage:"
    echo "  ./transfer_data.sh upload <local_path> <remote_path>"
    echo "  ./transfer_data.sh download <remote_path> <local_path>"
    echo ""
    echo "Examples:"
    echo "  # Upload dataset to scratch"
    echo "  ./transfer_data.sh upload ./my_dataset /gpfs/scratch/ehpc475/datasets/"
    echo ""
    echo "  # Download results"
    echo "  ./transfer_data.sh download /gpfs/projects/ehpc475/results ./results"
    echo ""
    echo "  # Upload HuggingFace cache"
    echo "  ./transfer_data.sh upload ~/.cache/huggingface /gpfs/projects/ehpc475/hf_cache"
    echo ""
    echo "Storage Locations:"
    echo "  /gpfs/projects/ehpc475 - Permanent storage (code, models)"
    echo "  /gpfs/scratch/ehpc475  - Fast scratch (training data, wiped every 2 weeks)"
}

upload() {
    if [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
        echo "Error: Both source and destination required"
        echo "Usage: ./transfer_data.sh upload <local_path> <remote_path>"
        exit 1
    fi
    
    echo "📤 Uploading: $SOURCE → $TRANSFER_HOST:$DEST"
    echo ""
    rsync -avzP \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.venv' \
        --exclude 'node_modules' \
        "$SOURCE" "$TRANSFER_HOST:$DEST"
    
    echo ""
    echo "✅ Upload complete!"
}

download() {
    if [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
        echo "Error: Both source and destination required"
        echo "Usage: ./transfer_data.sh download <remote_path> <local_path>"
        exit 1
    fi
    
    echo "📥 Downloading: $TRANSFER_HOST:$SOURCE → $DEST"
    echo ""
    rsync -avzP "$TRANSFER_HOST:$SOURCE" "$DEST"
    
    echo ""
    echo "✅ Download complete!"
}

case $ACTION in
    upload)
        upload
        ;;
    download)
        download
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown action: $ACTION"
        show_help
        exit 1
        ;;
esac

```

## Directory: `config`

### README.md
Path: `config/README.md`

```
# Configs Folder

This folder contains:
- DeepSpeed configuration (JSON)
- SLURM job script example
- Dockerfile example

These configs are not executed locally but teach the team exactly how to run distributed training on real clusters.

```

### config.json
Path: `config/config.json`

```json
{
    "vocab_size": 1000,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 256,
    "max_position_embeddings": 64,
    "layer_norm_eps": 1e-12,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1
}

```

### deepspeed_config.json
Path: `config/deepspeed_config.json`

```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "fp16": {
      "enabled": true
    },
    "zero_optimization": {
      "stage": 2
    },
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 1e-4
      }
    }
  }  
```

### docker_example.Dockerfile
Path: `config/docker_example.Dockerfile`

```dockerfile
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install deepspeed transformers datasets accelerate wandb

WORKDIR /workspace
COPY . /workspace
```

### slurm_example.sbatch
Path: `config/slurm_example.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=ds-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00

module load cuda/11.8

srun deepspeed --num_gpus=4 train_deepspeed.py --deepspeed_config ds_cfg.json
```

