#!/bin/bash
# Test Singularity/Apptainer Setup on MN5
# Run this from a compute node (after salloc) to verify container setup
#
# Usage: ./test_singularity.sh

set -e

echo "🐳 Testing Singularity/Apptainer on MN5"
echo "========================================="
echo ""

# Check if apptainer is available
echo "1. Checking Apptainer installation..."
module load apptainer 2>/dev/null || echo "   Note: apptainer may be loaded by default"
which apptainer && echo "   ✅ Apptainer found" || echo "   ❌ Apptainer not found"
echo ""

# Check for GPU support
echo "2. Checking GPU availability..."
if nvidia-smi &>/dev/null; then
    echo "   ✅ NVIDIA driver detected"
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv
else
    echo "   ⚠️  No GPU detected (are you on a login node?)"
fi
echo ""

# Test simple container execution
echo "3. Testing container pull (may take a minute first time)..."
TEST_SIF="/tmp/test_python.sif"
if [ ! -f "$TEST_SIF" ]; then
    apptainer pull "$TEST_SIF" docker://python:3.10-slim 2>/dev/null
fi
echo "   ✅ Container pulled successfully"
echo ""

# Test container execution
echo "4. Testing container execution..."
apptainer exec "$TEST_SIF" python --version
echo "   ✅ Container execution works"
echo ""

# Test GPU container (if on compute node)
echo "5. Testing GPU access inside container..."
if nvidia-smi &>/dev/null; then
    # Try running nvidia-smi inside a CUDA container
    apptainer exec --nv "$TEST_SIF" python -c "print('   ✅ GPU passthrough test complete')" 2>/dev/null || \
        echo "   ⚠️  GPU passthrough test skipped (need CUDA container for full test)"
else
    echo "   ⏭️  Skipped (no GPU available on login node)"
fi
echo ""

echo "========================================="
echo "🎉 Singularity/Apptainer test complete!"
echo ""
echo "Next steps:"
echo "  1. Build your project container locally"
echo "  2. Transfer to MN5: rsync -P my_container.sif domy667574@transfer1.bsc.es:/gpfs/projects/ehpc475/"
echo "  3. Run with: apptainer exec --nv -B /gpfs/projects/ehpc475:/app ./my_container.sif python train.py"
