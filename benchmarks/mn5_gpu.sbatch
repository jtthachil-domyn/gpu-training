#!/bin/bash
#SBATCH --job-name=bench_gpu
#SBATCH --account=ehpc475
#SBATCH --output=bench_gpu_%j.out
#SBATCH --error=bench_gpu_%j.err
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             # 1x H100
#SBATCH --cpus-per-task=20       # Required ratio
#SBATCH --time=00:10:00

module purge
# Load System PyTorch and all dependencies
module load gcc/11.4.0 mkl/2024.0 nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 hdf5/1.14.1-2-gcc python/3.11.5-gcc nccl/2.19.4 pytorch/2.4.0

# Venv not needed for system torch (psutil is optional and will be skipped)
# source ~/jthachil/bench_env/bin/activate

echo "=== Running MN5 GPU Benchmark (H100) ==="
python3 benchmark_suite.py --device cuda --matmul_size 16384 --out mn5_gpu_results.json
# Note: Matmul size increased to 16384 for H100 to actually make it sweat slightly
