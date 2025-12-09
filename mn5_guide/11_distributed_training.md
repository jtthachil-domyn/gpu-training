# 11. Distributed Training (AI/Deep Learning)

MareNostrum 5 (ACC partition) is designed for massive scale distributed training. However, the default environment often needs tweaking to make PyTorch/DeepSpeed happy with the underlying InfiniBand network.

## 1. The Hardware Reality
*   **Interconnect**: NVIDIA Quantum-2 InfiniBand (NDR).
*   **P2P**: NVLink between GPUs on the same node.
*   **Network**: InfiniBand between nodes.

## 2. PyTorch DDP Setup
Use the helper script `mn5_distributed_setup.py` provided in `examples/`.

### Key Environment Variables
MN5 requires explicit variables to tell NCCL which network interface to use. If you don't set these, NCCL might try to send tensors over the Ethernet management interface (slow/broken).

```bash
# In your SLURM script:

# 1. Force NCCL to use InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
# export NCCL_SOCKET_IFNAME=ib0 # Sometimes needed if auto-detection fails

# 2. Debugging (If training hangs)
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
```

## 3. DeepSpeed Configuration
When using DeepSpeed, rely on `hostfile` generation.

```bash
# Generate DeepSpeed hostfile from SLURM
scontrol show hostnames $SLURM_JOB_NODELIST > ./hostfile
```

Modify your `ds_config.json` to match the hardware:
```json
{
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true
  }
}
```

## 4. Common Issues
### "Watchdog caught hanging process"
**Cause**: IO Starvation. The GPUs are waiting for data from GPFS.
**Fix**:
1. Follow `mn5_dataloader.py` example (Pin memory, more workers).
2. Move data to `/gpfs/scratch` (NVMe-like speed).
3. Increase NCCL timeout: `export NCCL_ASYNC_ERROR_HANDLING=1` and `export TORCH_DISTRIBUTED_DEBUG=DETAIL`.

### "NCCL WARN: Connect to ... failed"
**Cause**: The nodes cannot talk to each other on the specified port.
**Fix**:
1. Ensure `MASTER_ADDR` is correct (see `mn5_distributed_setup.py`).
2. Ensure you are not blocked by firewall (use internal InfiniBand IPs, usually handled by hostnames).

## 5. Verification
Run the `examples/test_gpu_connectivity.py` script to benchmark:
1.  **Intra-node**: NVLink bandwidth (~400GB/s).
2.  **Inter-node**: InfiniBand bandwidth.
