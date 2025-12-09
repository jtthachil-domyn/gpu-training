import torch
import os
import time

def test_gpu():
    print(f"=== GPU Connectivity Test on {os.uname().node} ===")
    
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available.")
        return

    count = torch.cuda.device_count()
    print(f"[INFO] GPU Count: {count}")
    
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} | Mem: {props.total_memory / 1e9:.2f} GB")

    if count > 1:
        print("\n=== P2P Bandwidth Test (Approx) ===")
        # Simple test: Send tensor from GPU 0 to GPU 1
        t0 = torch.zeros(1024*1024*100, device='cuda:0') # 400MB float32
        torch.cuda.synchronize()
        
        start = time.time()
        t1 = t0.to('cuda:1')
        torch.cuda.synchronize()
        end = time.time()
        
        size_gb = (t0.nelement() * 4) / 1e9
        speed = size_gb / (end - start)
        print(f"[INFO] Transfer 0 -> 1: {speed:.2f} GB/s (Includes overhead)")
        print("Note: NVLink should be >200 GB/s. PCIe is ~20 GB/s.")
        
if __name__ == "__main__":
    test_gpu()
