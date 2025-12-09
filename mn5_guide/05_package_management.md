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
