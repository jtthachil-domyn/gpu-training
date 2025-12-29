# MN5 GPU Benchmarking Guide (Team Edition)

> **Goal:** Run PyTorch jobs on MareNostrum 5 (MN5) H100 GPUs without getting stuck in dependency hell or queue limits.

---

## 1. The Golden Rules (Read First)

1.  **NO Internet on Login Nodes**: You cannot `pip install` from PyPI directly.
    *   **Solution**: Use the **System Modules** (recommended) or download `.whl` files deeply offline.
2.  **Resource Ratio**: On ACC (GPU) nodes, you **MUST** request **20 CPUs per 1 GPU**.
    *   Example: `--gres=gpu:1 --cpus-per-task=20` (Correct)
    *   Example: `--gres=gpu:1 --cpus-per-task=10` (Will wait forever)
3.  **Use `bsc_acct`**: To check how many hours we have left.

---

## 2. Quick Setup (One-Time)

### SSH Configuration

Add this to your `~/.ssh/config` file on your Mac:

```bash
# MN5 General Purpose Login (CPU jobs)
Host mn5-gpp
    HostName glogin1.bsc.es
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes

# MN5 Accelerated Login (GPU H100 jobs)
Host mn5-acc
    HostName alogin1.bsc.es
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_rsa
    ForwardAgent yes

# MN5 Transfer Node (file transfers only)
Host mn5-transfer
    HostName transfer1.bsc.es
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_rsa
```

Replace `YOUR_USERNAME` with your assigned username (e.g., `domy944409`).

### Generate SSH Key (if you don't have one)

```bash
# Generate SSH key pair
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Copy public key to MN5 (you'll need your password)
ssh-copy-id YOUR_USERNAME@alogin1.bsc.es
```

### First Login

```bash
# SSH into the GPU login node (using the config alias)
ssh mn5-acc

# Or directly:
ssh YOUR_USERNAME@alogin1.bsc.es

# Create your personal workspace
mkdir ~/yourname
cd ~/yourname
```

---

## 3. The "Magic" Module Chain

MN5 dependencies are strict. Copy-paste this EXACT command to load a working environment.

**What does this do?**
1.  **`module purge`**: Clears your environment variables. **Safe to run:** It only affects your current terminal window, not other users or your other sessions.
2.  **`module load ...`**: Loads GCC, CUDA, **Python 3.11**, and **PyTorch 2.4** all at once.

```bash
module purge
module load gcc/11.4.0 mkl/2024.0 nvidia-hpc-sdk/23.11-cuda11.8 \
            openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 \
            impi/2021.11 hdf5/1.14.1-2-gcc python/3.11.5-gcc \
            nccl/2.19.4 pytorch/2.4.0
```

> **CRITICAL: Use a Virtual Environment (`venv`)**
> Even though the system provides Python and PyTorch, **ALWAYS** create a virtual environment (`venv`) to keep your specific project dependencies isolated.
>
> ```bash
> # Create venv using the system python we just loaded
> python3 -m venv .venv
> source .venv/bin/activate
> ```
> This prevents version conflicts between different projects you might work on.
> **Best Practice:** Don't put this directly in `.bashrc` (it can break other jobs). instead, use an alias.
>
> *   **If using account `domy667574`**: Just type `bench_mn5` (it is already set up).
> *   **If setting up a new account**: You can name the alias anything you want (e.g., `my_gpu_env`).
>
> 1.  **Add Alias**: Run this once (change `bench_mn5` to your preferred name):
>     ```bash
>     echo "alias bench_mn5='module purge; module load gcc/11.4.0 mkl/2024.0 nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 hdf5/1.14.1-2-gcc python/3.11.5-gcc nccl/2.19.4 pytorch/2.4.0'" >> ~/.bashrc
>     source ~/.bashrc
>     ```
>
> 2.  **Usage**: Just type `bench_mn5` (or your chosen name) when you log in.
> 3.  **List Aliases**: Type `alias` to see all defined shortcuts.
---

## 4. Workflow: Develop Locally -> Run Remotely

### Step A: Sync Code (From Mac)
We edit code on our Macs and `rsync` it up.

```bash
# Run this on your MAC
# Replace 'benchmarks/' with your local folder name
rsync -avz benchmarks/ domy667574@alogin1.bsc.es:~/yourname/
```

### Step B: Submit Jobs (From MN5)
Use `sbatch` to submit jobs to the queue. **Do not run heavy training on the login node!**

```bash
# Run this on MN5
cd ~/yourname

# Submit files (ensure #SBATCH headers are correct)
sbatch mn5_cpu.sbatch
sbatch mn5_gpu.sbatch
```

### Step C: Monitor
```bash
sq                         # Check your active jobs
bsc_acct                   # Check remaining budget
cat job_name_12345.err     # Check errors (if job crashes)
```

### Step D: Get Results (From Mac)
Download the output files back to your Mac for analysis.

```bash
# Run on MAC
# Note the quotes "" around the path to handle wildcards!
rsync -avz "domy667574@alogin1.bsc.es:~/yourname/*.json" ./results/
```

---

## 5. Installing Additional Packages (Offline)

Since MN5 has no internet, use this workflow to install packages like `transformers`:

### On Your Mac (with internet):
```bash
mkdir ~/mn5_wheels && cd ~/mn5_wheels

# Download wheels for Python 3.11 + Linux
pip3 download --no-deps transformers accelerate huggingface-hub tokenizers \
    safetensors tqdm fsspec pyyaml regex numpy packaging requests \
    --python-version 3.11 --platform manylinux2014_x86_64 --only-binary=:all:

# Transfer to MN5
rsync -avz ~/mn5_wheels/ your_user@transfer1.bsc.es:~/wheels/
```

### On MN5:
```bash
# Load modules and activate venv
your_alias_name
source ~/your_project/venv/bin/activate

# Install from local wheels
pip install --no-index --find-links ~/wheels/ transformers accelerate
```

---

## 6. Singularity Containers (Recommended for Portability)

For long-term portability across different systems (T4 VM, H100 HPC, etc.), use **Singularity containers**.

### Step 1: Start from Official TRL Dockerfile

Create `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip uv
RUN uv pip install --system trl[liger,peft,vlm] kernels trackio
```

### Step 2: Build Docker Image (on Mac)
```bash
# Build for Linux/AMD64 architecture
docker build --platform linux/amd64 -t trl-train:latest .
```

### Step 3: Convert to Singularity
```bash
# Save Docker image as tar
docker save trl-train:latest -o trl-train.tar

# Convert to Singularity (on a Linux machine or MN5)
singularity build trl-train.sif docker-archive://trl-train.tar
```

### Step 4: Transfer to MN5

Use `scp` or `rsync` to transfer the tarball to the transfer node.

```bash
# Option 1: SCP (Simple)
scp trl-train.tar YOUR_USERNAME@transfer1.bsc.es:~/

# Option 2: Rsync (Better for resuming)
rsync -avz --progress trl-train.tar YOUR_USERNAME@transfer1.bsc.es:~/
```

> **Note:** If transfers keep failing with "Connection reset", your internet might be unstable. See **Troubleshooting** for the "Split File" workaround.

### Step 5: Build on MN5

**Critical Fix:** You must manually create the session directory structure first.

```bash
# 1. Create session directories (Run once)
mkdir -p /scratch/tmp/singularity/mnt/session

# 2. Build SIF image
module load singularity/4.1.5
singularity build trl-train.sif docker-archive://trl-train.tar
```

### Step 6: Run
```bash
singularity exec --nv trl-train.sif python your_script.py
```

> **Why Singularity?**
> - Works identically on T4 VM and H100 HPC
> - No module chain management
> - Reproducible environment
> - Easy to share with team members

---

## 7. Troubleshooting

*   **"Connection reset by peer" during transfer**: This usually means unstable internet or server timeout. **Workaround:** Split the file:
    ```bash
    # 1. Split into 30MB chunks
    split -b 30M trl-train.tar trl-train.tar.part_
    # 2. Loop transfer
    for p in trl-train.tar.part_*; do scp "$p" user@transfer1.bsc.es:~/; done
    # 3. Join on MN5
    cat trl-train.tar.part_* > trl-train.tar
    ```
*   **"failed to resolve session directory"**: You need to run `mkdir -p /scratch/tmp/singularity/mnt/session`.
*   **"Socket verification failed"**: Your SSH agent is acting up. Run `ssh-add -D` and try again.
*   **"Requested node configuration is not available"**: You violated the 1:20 GPU:CPU ratio. Fix your `.sbatch` file.
*   **"ModuleNotFoundError: No module named torch"**: You forgot to load the magic module chain (Step 3) in your script or interactive session.
*   **"No module named 'tf_keras'"**: TensorFlow/Keras conflict. Use specific imports: `from transformers import AutoModel` instead of `from transformers import *`.
*   **Flash Attention 2 errors on T4**: T4 (SM 7.5) doesn't support FA2. Use `attn_implementation="sdpa"` or run on H100.
