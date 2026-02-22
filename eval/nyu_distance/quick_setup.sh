#!/bin/bash
# Quick setup for NYU Distance Evaluation
#
# The full NYU dataset is 2.8GB. This script provides options:
# 1. Download full dataset (slow but complete)
# 2. Use pre-existing data if available
# 3. Skip and run with --skip_download flag

set -e

echo "======================================================================"
echo "NYU DEPTH V2 - QUICK SETUP"
echo "======================================================================"
echo ""

NYU_DIR="data/nyu_depth_v2"
EXTRACTED_DIR="$NYU_DIR/extracted"
MAT_FILE="$NYU_DIR/nyu_depth_v2_labeled.mat"

# Check if extracted data already exists
if [ -d "$EXTRACTED_DIR/rgb" ] && [ "$(ls -A $EXTRACTED_DIR/rgb 2>/dev/null)" ]; then
    echo "✓ NYU dataset already extracted: $EXTRACTED_DIR"
    echo "  Found $(ls $EXTRACTED_DIR/rgb | wc -l) RGB images"
    echo ""
    echo "Ready to run evaluation!"
    echo "  python eval/nyu_distance/eval_nyu_distance.py"
    exit 0
fi

# Check if .mat file exists
if [ -f "$MAT_FILE" ]; then
    echo "✓ Found dataset file: $MAT_FILE"
    echo "  Extracting samples..."
    python data/nyu_depth_v2/download_nyu.py
    exit 0
fi

echo "Dataset not found."
echo ""
echo "OPTIONS:"
echo "  1. Download full dataset (~2.8GB, requires ~10 min)"
echo "  2. Run without dataset (will fail, for testing pipeline only)"
echo ""
read -p "Enter choice [1/2]: " choice

case $choice in
    1)
        echo ""
        echo "Downloading NYU Depth V2..."
        python data/nyu_depth_v2/download_nyu.py
        ;;
    2)
        echo ""
        echo "Skipping download. You can run with --skip_download flag."
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
