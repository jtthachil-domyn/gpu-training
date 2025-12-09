import os
import sys
import time
import argparse
import socket

def simulate_barrier(name):
    print(f"[{name}] Entering barrier...")
    time.sleep(1) # Simulate network sync delay
    print(f"[{name}] Exited barrier.")

def main():
    parser = argparse.ArgumentParser(description="Fake Multi-Node Launcher Simulation")
    # We can accept args or read env vars. Docs emphasize env vars.
    parser.parse_args()

    print("--- FAKE MULTI-NODE LAUNCHER SIMULATION ---")
    
    # 1. Read Environment Variables
    # These are typically set by SLURM or a launcher script (like torchrun)
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size_nodes = int(os.environ.get("WORLD_SIZE_NODES", "1")) # Number of nodes
    nproc_per_node = int(os.environ.get("NPROC_PER_NODE", "4")) # GPUs per node
    
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    # 2. Calculate Global Rank (The Mapping Logic)
    # global_rank = (node_rank * nproc_per_node) + local_rank
    # total_world_size = world_size_nodes * nproc_per_node
    
    global_rank = (node_rank * nproc_per_node) + local_rank
    total_world_size = world_size_nodes * nproc_per_node
    
    # In a real scenario, WORLD_SIZE is usually the total world size already.
    # But for this simulation, we are deriving it to show understanding.
    
    identity = f"Node {node_rank} | Local Rank {local_rank}"

    print(f"\n[Environment Configuration]")
    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")
    print(f"Nodes: {world_size_nodes}")
    print(f"Procs per Node: {nproc_per_node}")
    
    print(f"\n[Identity Reading]")
    print(f"NODE_RANK  : {node_rank}")
    print(f"LOCAL_RANK : {local_rank}")
    
    print(f"\n[Calculated Global State]")
    print(f"GLOBAL_RANK: {global_rank}")
    print(f"WORLD_SIZE : {total_world_size}")
    
    print(f"\n[{identity}] Initializing Fake Process Group...")
    time.sleep(0.5)
    print(f"[{identity}] Connected to {master_addr}:{master_port}")
    
    # 3. Simulate Distributed Barrier
    print(f"\n[{identity}] Simulating Distributed Barrier...")
    simulate_barrier(identity)
    
    # 4. SLURM Concept Explanation
    print(f"\n[{identity}] SLURM Context (Concept):")
    print(f"  If running under SLURM:")
    print(f"  SLURM_PROCID would be: {global_rank}")
    print(f"  SLURM_LOCALID would be: {local_rank}")
    print(f"  SLURM_NODEID  would be: {node_rank}")
    
    # 5. Simulate Training Step
    print(f"\n[{identity}] Starting Fake Training Step 1...")
    time.sleep(0.5)
    print(f"[{identity}] Forward pass complete.")
    print(f"[{identity}] Backward pass (Gradient All-Reduce) complete.")
    print(f"[{identity}] Optimizer step complete.")
    
    print(f"\n[{identity}] Simulation Complete. Exiting.")

if __name__ == "__main__":
    main()