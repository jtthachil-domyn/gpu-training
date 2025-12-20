import argparse
import time
import torch
import json
import os
import platform


def get_system_info():
    info = {
        "system": platform.system(),
        "processor": platform.processor(),
    }
    
    # Try to get detailed stats (Works on Local Mac)
    # Fails gracefully on MN5 if psutil is missing
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        info["cpu_count_physical"] = "Unknown (psutil missing)"
        info["ram_gb"] = "Unknown"
        
    return info

def benchmark_matmul(device, size=8192, dtype=torch.float32):
    print(f"--- Benchmarking Matrix Multiplication ({size}x{size}) on {device} ---")
    
    try:
        if device == "cuda":
            if not torch.cuda.is_available():
                print("CUDA not available, skipping.")
                return None
            t_device = torch.device("cuda")
        elif device == "mps":
            if not torch.backends.mps.is_available():
                print("MPS not available, skipping.")
                return None
            t_device = torch.device("mps")
        else:
            t_device = torch.device("cpu")

        # Create matrices
        a = torch.randn(size, size, dtype=dtype, device=t_device)
        b = torch.randn(size, size, dtype=dtype, device=t_device)

        # Warmup
        print("Warming up...")
        for _ in range(2):
            c = torch.matmul(a, b)
        
        if device == "cuda": torch.cuda.synchronize()
        if device == "mps": torch.mps.synchronize()

        # Benchmark
        print("Running benchmark (5 iterations)...")
        start_time = time.time()
        for _ in range(5):
            c = torch.matmul(a, b)
            if device == "cuda": torch.cuda.synchronize()
            if device == "mps": torch.mps.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / 5
        # TFLOPS: 2 * N^3 operations
        ops = 2 * (size ** 3)
        tflops = (ops / avg_time) / 1e12

        print(f"Average Time: {avg_time:.4f} s")
        print(f"Performance: {tflops:.4f} TFLOPS")
        
        return {
            "size": size,
            "avg_time_seconds": avg_time,
            "tflops": tflops
        }
    except Exception as e:
        print(f"Error during matmul: {e}")
        return None

def benchmark_bandwidth(device, size_elem=100_000_000, dtype=torch.float32):
    print(f"\n--- Benchmarking Memory Bandwidth (Vector Add, {size_elem} elems) on {device} ---")
    
    try:
        if device == "cuda":
            if not torch.cuda.is_available(): return None
            t_device = torch.device("cuda")
        elif device == "mps":
            if not torch.backends.mps.is_available(): return None
            t_device = torch.device("mps")
        else:
            t_device = torch.device("cpu")

        # Create vectors (size_elem * 4 bytes for float32)
        # We need A, B, C. Total memory = 3 * size * 4 bytes
        # 100M float32 = 400MB. Total 1.2GB.
        
        a = torch.randn(size_elem, dtype=dtype, device=t_device)
        b = torch.randn(size_elem, dtype=dtype, device=t_device)

        # Warmup
        c = a + b
        if device == "cuda": torch.cuda.synchronize()
        if device == "mps": torch.mps.synchronize()

        # Benchmark
        print("Running benchmark (10 iterations)...")
        start_time = time.time()
        for _ in range(10):
            c = a + b
            if device == "cuda": torch.cuda.synchronize()
            if device == "mps": torch.mps.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        
        # Read A, Read B, Write C = 3 * size bytes
        total_bytes = 3 * size_elem * 4 
        gb_per_sec = (total_bytes / avg_time) / 1e9

        print(f"Average Time: {avg_time:.6f} s")
        print(f"Bandwidth: {gb_per_sec:.2f} GB/s")

        return {
            "size_elements": size_elem,
            "avg_time_seconds": avg_time,
            "bandwidth_gb_s": gb_per_sec
        }
    except Exception as e:
        print(f"Error during bandwidth: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], default="auto")
    parser.add_argument("--matmul_size", type=int, default=8192)
    parser.add_argument("--out", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    # Auto-detect best device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Target Device: {device}")
    
    results = {
        "timestamp": time.time(),
        "device": device,
        "system_info": get_system_info(),
        "matmul": benchmark_matmul(device, size=args.matmul_size),
        "bandwidth": benchmark_bandwidth(device)
    }

    # Add Device Name info
    if device == "cuda":
        results["device_name"] = torch.cuda.get_device_name(0)
    elif device == "mps":
        results["device_name"] = "Apple Metal (MPS)"
    else:
        results["device_name"] = platform.processor()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {args.out}")

if __name__ == "__main__":
    main()
