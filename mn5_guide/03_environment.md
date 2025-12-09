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
