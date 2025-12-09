#!/bin/bash
source venv/bin/activate
set -e

echo "----------------------------------------------------------------"
echo "VERIFYING LAB 1: Tiny Transformer Training"
echo "----------------------------------------------------------------"
python scripts/train_tiny_transformer.py --epochs 1 --batch-size 4 --log-interval 1
if [ $? -eq 0 ]; then
    echo "Lab 1 Passed."
else
    echo "Lab 1 Failed."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "VERIFYING LAB 2: DDP Simulation"
echo "----------------------------------------------------------------"
torchrun --nproc_per_node=2 scripts/ddp_simulation_train.py --epochs 1 --batch-size 2
if [ $? -eq 0 ]; then
    echo "Lab 2 Passed."
else
    echo "Lab 2 Failed."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "VERIFYING LAB 3: Fake Multi-Node Launcher"
echo "----------------------------------------------------------------"
export NODE_RANK=0
export LOCAL_RANK=1
export WORLD_SIZE_NODES=2
export NPROC_PER_NODE=4
python scripts/fake_multi_node_launcher.py
if [ $? -eq 0 ]; then
    echo "Lab 3 Passed."
else
    echo "Lab 3 Failed."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "ALL LABS VERIFIED SUCCESSFULLY"
echo "----------------------------------------------------------------"
