#!/bin/bash
# Interactive GPU Session for Development
# Usage: ./interactive_session.sh [hours] [gpus]
#
# Examples:
#   ./interactive_session.sh           # 1 hour, 1 GPU (default)
#   ./interactive_session.sh 2         # 2 hours, 1 GPU
#   ./interactive_session.sh 2 4       # 2 hours, 4 GPUs

HOURS=${1:-1}
GPUS=${2:-1}
CPUS=$((GPUS * 20))  # 20 CPUs per GPU rule

echo " Requesting interactive session..."
echo "   Duration: ${HOURS} hour(s)"
echo "   GPUs: ${GPUS}"
echo "   CPUs: ${CPUS}"
echo ""

salloc \
    --account=ehpc475 \
    --partition=acc \
    --qos=acc_debug \
    --gres=gpu:${GPUS} \
    --cpus-per-task=${CPUS} \
    --time=${HOURS}:00:00

# After allocation, you'll be on a compute node
# Run: nvidia-smi to verify GPU access
# Run: python train.py to test your code
