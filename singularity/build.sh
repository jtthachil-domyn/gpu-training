#!/bin/bash
# Build and deploy Singularity container for MN5 testing
# Run this on your Mac

set -e

echo "=== Step 1: Building Docker image ==="
docker build --platform linux/amd64 -t mn5-test:latest -f Dockerfile.test .

echo "=== Step 2: Saving Docker image as tar ==="
docker save mn5-test:latest -o mn5-test.tar

echo "=== Step 3: Converting to Singularity ==="
# Note: This step requires Singularity installed locally OR do it on MN5
# If you have Singularity locally:
# singularity build mn5-test.sif docker-archive://mn5-test.tar
# Otherwise, transfer the tar to MN5 and convert there

echo "=== Docker image saved to mn5-test.tar ==="
echo ""
echo "Next steps:"
echo "1. Transfer to MN5: rsync -avz mn5-test.tar your_user@transfer1.bsc.es:~/"
echo "2. On MN5, convert to Singularity:"
echo "   singularity build mn5-test.sif docker-archive://mn5-test.tar"
echo "3. Run test:"
echo "   singularity run mn5-test.sif"
