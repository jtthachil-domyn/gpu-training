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
