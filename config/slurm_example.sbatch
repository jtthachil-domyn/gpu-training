#!/bin/bash
#SBATCH --job-name=ds-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00

module load cuda/11.8

srun deepspeed --num_gpus=4 train_deepspeed.py --deepspeed_config ds_cfg.json