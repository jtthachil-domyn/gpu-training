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
