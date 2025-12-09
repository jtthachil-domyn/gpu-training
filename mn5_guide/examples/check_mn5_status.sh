#!/bin/bash
# A quick health-check script for your MN5 account

echo "=== 1. Checking Quota  ==="
bsc_quota
echo ""

echo "=== 2. Active Jobs ==="
squeue -u $USER
echo ""

echo "=== 3. Login Nodes Load ==="
bsc_load
echo ""

echo "=== 4. File Count (Inodes) ==="
# Approximated. Real check uses bsc_quota but sometimes this is useful.
echo "Checking $HOME..."
find $HOME -maxdepth 1 | wc -l
