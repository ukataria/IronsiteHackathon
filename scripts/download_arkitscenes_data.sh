#!/bin/bash
# Download ARKitScenes validation data for benchmarking
#
# This script downloads a subset of the ARKitScenes validation split
# with RGB (lowres_wide) and depth (lowres_depth) data.

set -e

echo "============================================================"
echo "ARKitScenes Data Download"
echo "============================================================"
echo ""

# Check if ARKitScenes repo exists
if [ ! -d "ARKitScenes" ]; then
    echo "Cloning ARKitScenes repository..."
    git clone https://github.com/apple/ARKitScenes.git
    cd ARKitScenes
else
    echo "ARKitScenes repository already exists"
    cd ARKitScenes
    git pull
fi

echo ""
echo "Downloading validation split (RGB + depth)..."
echo "This may take 10-30 minutes depending on your connection"
echo ""

# Download validation data with RGB and depth
# Using the raw split with lowres_wide (RGB) and lowres_depth assets
python3 download_data.py raw \
    --split Validation \
    --video_id_csv raw/raw_train_val_splits.csv \
    --download_dir ../data/raw/ARKitScenes \
    --raw_dataset_assets lowres_wide lowres_depth lowres_wide_intrinsics

echo ""
echo "============================================================"
echo "Download complete!"
echo "============================================================"
echo ""
echo "Data location: data/raw/ARKitScenes/Validation/"
echo ""
echo "Next steps:"
echo "1. Convert the downloaded data to benchmark format:"
echo "   python scripts/convert_arkitscenes_to_benchmark.py"
echo ""
echo "2. Run the benchmark:"
echo "   python eval/arkit/benchmark_arkit_all_models.py \\"
echo "     --arkit_data_dir data/arkit \\"
echo "     --num_images 10 \\"
echo "     --pairs_per_image 3"
echo ""
