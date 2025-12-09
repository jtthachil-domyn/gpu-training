import os
import subprocess
import torch
import torch.distributed as dist

def setup_mn5_distributed():
    """
    Robustly configures the PyTorch Distributed environment for MareNostrum 5.
    Handles specific SLURM environment variables and sets up the MASTER_ADDR/PORT.
    """
    if "SLURM_JOB_ID" not in os.environ:
        print("[WARN] Not running inside SLURM. Defaulting to local standalone mode.")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        dist.init_process_group(backend="nccl")
        return

    # 1. Get List of Nodes
    # MN5 uses a compressed format like "node[01-04]". We need the first one as master.
    try:
        nodelist = os.environ.get("SLURM_JOB_NODELIST")
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist]
        ).decode().strip().splitlines()
    except Exception as e:
        raise RuntimeError(f"Could not parse SLURM nodelist: {e}")

    master_node = hostnames[0]
    
    # 2. Set Environment Variables for Torch
    os.environ["MASTER_ADDR"] = master_node
    # Pick a random port or a fixed one. 
    # Just ensure it doesn't conflict if multiple jobs share a node (unlikely in exclusive mode).
    os.environ["MASTER_PORT"] = "29500" 

    # SLURM_PROCID is the global rank (0 to Total GPUs - 1)
    os.environ["RANK"] = os.environ.get("SLURM_PROCID")
    
    # SLURM_NTASKS is the world size
    os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS")
    
    # SLURM_LOCALID is the local rank on the node (0, 1, 2, 3)
    os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID")

    print(f"[MN5-Setup] Master: {master_node}, Rank: {os.environ['RANK']}, World: {os.environ['WORLD_SIZE']}")

    # 3. Initialize Process Group
    # 'nccl' is mandatory for H100s.
    dist.init_process_group(backend="nccl")
    
    # 4. Set Device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    print(f"[MN5-Setup] Successfully initialized rank {os.environ['RANK']} on {os.uname().node}")

if __name__ == "__main__":
    setup_mn5_distributed()
    # Your training loop here
