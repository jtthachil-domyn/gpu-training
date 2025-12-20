#!/bin/bash
# Local Benchmark Runner (Mac)

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"

# Activate VENV
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    echo "Activating venv..."
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: venv not found at $PROJECT_ROOT/venv/bin/activate"
fi

cd "$DIR"

echo "=== Running Local Benchmark (MPS/Metal) ==="
# Check if MPS is available
if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)"; then
    python3 benchmark_suite.py --device mps --out local_mps_results.json
else
    echo "MPS not available on this environment."
fi

echo ""
echo "=== Running Local Benchmark (CPU) ==="
python3 benchmark_suite.py --device cpu --matmul_size 4096 --out local_cpu_results.json

echo ""
echo "Done."
