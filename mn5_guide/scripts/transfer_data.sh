#!/bin/bash
# Data Transfer Helper for MN5
# Use this script to transfer files to/from MN5 via the transfer nodes
#
# Usage:
#   ./transfer_data.sh upload /local/path /remote/path
#   ./transfer_data.sh download /remote/path /local/path
#
# Examples:
#   ./transfer_data.sh upload ./my_dataset /gpfs/scratch/ehpc475/datasets
#   ./transfer_data.sh download /gpfs/projects/ehpc475/results ./results

set -e

TRANSFER_HOST="domy667574@transfer1.bsc.es"
ACTION=${1:-help}
SOURCE=${2:-}
DEST=${3:-}

show_help() {
    echo "MN5 Data Transfer Helper"
    echo "========================"
    echo ""
    echo "Usage:"
    echo "  ./transfer_data.sh upload <local_path> <remote_path>"
    echo "  ./transfer_data.sh download <remote_path> <local_path>"
    echo ""
    echo "Examples:"
    echo "  # Upload dataset to scratch"
    echo "  ./transfer_data.sh upload ./my_dataset /gpfs/scratch/ehpc475/datasets/"
    echo ""
    echo "  # Download results"
    echo "  ./transfer_data.sh download /gpfs/projects/ehpc475/results ./results"
    echo ""
    echo "  # Upload HuggingFace cache"
    echo "  ./transfer_data.sh upload ~/.cache/huggingface /gpfs/projects/ehpc475/hf_cache"
    echo ""
    echo "Storage Locations:"
    echo "  /gpfs/projects/ehpc475 - Permanent storage (code, models)"
    echo "  /gpfs/scratch/ehpc475  - Fast scratch (training data, wiped every 2 weeks)"
}

upload() {
    if [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
        echo "Error: Both source and destination required"
        echo "Usage: ./transfer_data.sh upload <local_path> <remote_path>"
        exit 1
    fi
    
    echo "📤 Uploading: $SOURCE → $TRANSFER_HOST:$DEST"
    echo ""
    rsync -avzP \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.venv' \
        --exclude 'node_modules' \
        "$SOURCE" "$TRANSFER_HOST:$DEST"
    
    echo ""
    echo "✅ Upload complete!"
}

download() {
    if [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
        echo "Error: Both source and destination required"
        echo "Usage: ./transfer_data.sh download <remote_path> <local_path>"
        exit 1
    fi
    
    echo "📥 Downloading: $TRANSFER_HOST:$SOURCE → $DEST"
    echo ""
    rsync -avzP "$TRANSFER_HOST:$SOURCE" "$DEST"
    
    echo ""
    echo "✅ Download complete!"
}

case $ACTION in
    upload)
        upload
        ;;
    download)
        download
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown action: $ACTION"
        show_help
        exit 1
        ;;
esac
