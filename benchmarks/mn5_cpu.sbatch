#!/bin/bash
#SBATCH --job-name=bench_cpu
#SBATCH --account=ehpc475
#SBATCH --output=bench_cpu_%j.out
#SBATCH --error=bench_cpu_%j.err
#SBATCH --partition=acc          # Using ACC partition but running on CPU
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1             # Must request GPU to get on specific ACC node slice
#SBATCH --cpus-per-task=20       # 20 CPUs per GPU rule
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
# Remove exclusive to allow debug queue usage
# #SBATCH --exclusive

module purge
# Load System PyTorch and all dependencies
module load gcc/11.4.0 mkl/2024.0 nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 impi/2021.11 hdf5/1.14.1-2-gcc python/3.11.5-gcc nccl/2.19.4 pytorch/2.4.0

# Venv not needed for system torch (psutil is optional and will be skipped)
# source ~/jthachil/bench_env/bin/activate

echo "=== Running MN5 CPU Benchmark (Sapphire Rapids) ==="
# Force PyTorch to use CPU even if GPU is present
export CUDA_VISIBLE_DEVICES=""

# Execute using system python
python3 benchmark_suite.py --device cpu --matmul_size 8192 --out mn5_cpu_results.json
