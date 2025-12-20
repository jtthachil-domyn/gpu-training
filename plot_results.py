import json
import matplotlib.pyplot as plt
import os
import argparse

def load_result(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    files = {
        "Local Mac (MPS)": "benchmarks/local_mps_results.json",
        "Local Mac (CPU)": "benchmarks/local_cpu_results.json",
        "MN5 CPU (GPP)":   "benchmarks/mn5_cpu_results.json",
        "MN5 GPU (H100)":  "benchmarks/mn5_gpu_results.json"
    }
    
    data = {}
    for label, path in files.items():
        res = load_result(path)
        if res:
            data[label] = res

    if not data:
        print("No data found!")
        return

    # Prepare Data
    labels = list(data.keys())
    tflops = [d["matmul"]["tflops"] if d["matmul"] else 0 for d in data.values()]
    bandwidth = [d["bandwidth"]["bandwidth_gb_s"] if d["bandwidth"] else 0 for d in data.values()]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # TFLOPS
    ax1.bar(labels, tflops, color=['skyblue', 'lightgreen', 'orange', 'red'])
    ax1.set_title("Compute Performance (TFLOPS)")
    ax1.set_ylabel("TFLOPS (Higher is Better)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Bandwidth
    ax2.bar(labels, bandwidth, color=['skyblue', 'lightgreen', 'orange', 'red'])
    ax2.set_title("Memory Bandwidth")
    ax2.set_ylabel("GB/s (Higher is Better)")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("benchmark_comparison.png")
    print("Plot saved to benchmark_comparison.png")

if __name__ == "__main__":
    main()
