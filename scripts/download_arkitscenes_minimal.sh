#!/bin/bash
# Download minimal ARKitScenes data (1-2 scenes only)
#
# This downloads just enough data to test the benchmark (~500MB-1GB)
# instead of the full validation split (50-100GB)

set -e

echo "============================================================"
echo "ARKitScenes Minimal Data Download (1-2 scenes)"
echo "============================================================"
echo ""

# Specific scene IDs to download (small validation scenes)
SCENE_IDS=(
    "42445173"
    "42445677"
)

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
echo "Downloading ${#SCENE_IDS[@]} validation scenes (RGB + depth)..."
echo "Estimated size: ~500MB-1GB total"
echo ""

# Create a temporary CSV with the required format
# The download script expects 'video_id' and 'fold' columns
TEMP_CSV="temp_minimal_scenes.csv"
echo "video_id,fold" > $TEMP_CSV
for scene_id in "${SCENE_IDS[@]}"; do
    echo "$scene_id,Validation" >> $TEMP_CSV
done

# Download only these specific scenes
python3 download_data.py raw \
    --split Validation \
    --video_id_csv $TEMP_CSV \
    --download_dir ../data/raw/ARKitScenes \
    --raw_dataset_assets lowres_wide lowres_depth lowres_wide_intrinsics

# Clean up temp CSV
rm $TEMP_CSV

echo ""
echo "============================================================"
echo "Download complete!"
echo "============================================================"
echo ""
echo "Downloaded ${#SCENE_IDS[@]} scenes to: data/raw/ARKitScenes/Validation/"
echo ""
echo "Next steps:"
echo "1. Convert the downloaded data to benchmark format:"
echo "   python scripts/convert_arkitscenes_to_benchmark.py \\"
echo "     --source_dir data/raw/ARKitScenes/Validation \\"
echo "     --output_dir data/arkit \\"
echo "     --max_frames_per_scene 20 \\"
echo "     --max_scenes 2"
echo ""
echo "2. Run the benchmark:"
echo "   python eval/arkit/benchmark_arkit_all_models.py \\"
echo "     --arkit_data_dir data/arkit \\"
echo "     --num_images 10 \\"
echo "     --pairs_per_image 3"
echo ""
