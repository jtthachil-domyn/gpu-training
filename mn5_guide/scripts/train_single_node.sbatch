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
