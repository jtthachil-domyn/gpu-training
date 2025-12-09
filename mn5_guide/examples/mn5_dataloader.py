import torch
from torch.utils.data import DataLoader, Dataset
import tempfile
import os

class Mn5Dataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        # On MN5, ensure your data is in /gpfs/scratch, NOT /gpfs/projects or HOME.
        # Projects is optimized for throughput, not IOPS (latency).
        # Home has a strict quota and is slow.
        self.check_path()

    def check_path(self):
        cwd = os.getcwd()
        if "/gpfs/home" in cwd:
            print("[WARN] You are running from HOME. This will be slow and may hit quota limits.")
        elif "/gpfs/projects" in cwd:
            print("[INFO] Running from PROJECTS. Read-only large files are fine here.")
        elif "/gpfs/scratch" in cwd:
            print("[OK] Running from SCRATCH. Optimal for active IO.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate data loading
        return torch.randn(3, 224, 224), torch.randint(0, 100, (1,))

def get_optimized_loader(dataset, batch_size=32):
    """
    Returns a DataLoader optimized for MN5 hardware (112 cores per node).
    """
    
    # MN5 Rule: 1 GPU : 20 CPUs.
    # So you realistically have ~15-18 works available per GPU.
    # Setting num_workers too high (>20) creates overhead.
    # Setting it too low (0) causes GPU starvation.
    NUM_WORKERS = 16 
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # Key for GPFS:
        num_workers=NUM_WORKERS,
        # Keeps workers alive between epochs to avoid re-forking overhead
        persistent_workers=True, 
        # Crucial for GPU transfer speed
        pin_memory=True,
        # prefetch_factor * num_workers = batches buffered. 
        # Don't set too high or you scream OOM. 2 is usually safe.
        prefetch_factor=2 
    )
    return loader

if __name__ == "__main__":
    ds = Mn5Dataset()
    loader = get_optimized_loader(ds)
    print("DataLoader ready. Iterating...")
    for batch in loader:
        pass
    print("Done.")
