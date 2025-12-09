import os
import torch
import shutil
import time

def save_checkpoint_safe(model, optimizer, epoch, path):
    """
    Saves a checkpoint atomically to prevent corruption on GPFS.
    GPFS is robust, but if your job hits a time limit mid-write, 
    you end up with a corrupt, size-0 file.
    
    Strategy: Write to .tmp file, then rename. Rename is atomic.
    """
    
    # 1. Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 2. Define temp path
    tmp_path = path + ".tmp"
    
    print(f"[Checkpoint] Saving to temporary file: {tmp_path}")
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 3. Save to tmp
    try:
        torch.save(state, tmp_path)
    except Exception as e:
        print(f"[Error] Failed to save checkpoint: {e}")
        return

    # 4. Atomic Rename
    # If the job dies here, you either hava the old file OR the new file.
    # Never a half-written file.
    shutil.move(tmp_path, path)
    print(f"[Checkpoint] Successfully renamed to: {path}")

    # 5. Cleanup Old Checkpoints (Optional)
    # MN5 Scratch has a file count limit. Clean up!
    # implementation specific...

def load_checkpoint(path, model, optimizer):
    if not os.path.exists(path):
        print("[Checkpoint] No checkpoint found. Starting from scratch.")
        return 0
    
    print(f"[Checkpoint] Loading from {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
