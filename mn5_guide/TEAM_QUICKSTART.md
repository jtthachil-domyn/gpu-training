# Team Quickstart: MN5 for Humans (Detailed)

This guide bridges the gap between basic analogies and power-user workflows.

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
salloc --partition=acc --qos=acc_debug --gres=gpu:1 --cpus-per-task=20 --time=01:00:00 --account=bscXX
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

