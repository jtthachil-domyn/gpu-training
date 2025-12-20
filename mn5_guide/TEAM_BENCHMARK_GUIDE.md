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

ensure you have your SSH config set up in `~/.ssh/config` (ask Joseph for the snippet).

```bash
# SSH into the GPU login node
ssh domy667574@alogin1.bsc.es

# Create your personal workspace (replace 'yourname')
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

## 5. Troubleshooting

*   **"Socket verification failed"**: Your SSH agent is acting up. Run `ssh-add -D` and try again.
*   **"Requested node configuration is not available"**: You violated the 1:20 GPU:CPU ratio. Fix your `.sbatch` file.
*   **"ModuleNotFoundError: No module named torch"**: You forgot to load the magic module chain (Step 3) in your script or interactive session.
