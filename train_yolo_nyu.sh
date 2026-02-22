#!/bin/bash
#
# Fine-tune YOLO on NYU Dataset for Anchor Detection
#
# This script:
# 1. Converts NYU semantic labels to YOLO format
# 2. Trains YOLO on 11 anchor object classes
# 3. Saves fine-tuned model for two-head architecture

set -e

echo "=========================================="
echo "YOLO FINE-TUNING ON NYU DATASET"
echo "=========================================="
echo ""

# Check for NYU dataset
if [ ! -d "data/nyu_depth_v2/extracted" ]; then
    echo "✗ NYU dataset not found!"
    echo "  Run: python data/nyu_depth_v2/download_nyu.py"
    exit 1
fi

echo "✓ NYU dataset found"
echo ""

# Configuration
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
DEVICE=${DEVICE:-cuda}

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""

# Check CUDA
if [ "$DEVICE" = "cuda" ]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "⚠️  CUDA not available, falling back to CPU"
        DEVICE="cpu"
    fi
fi

# Run training
python train/finetune_yolo_nyu.py \
    --nyu_data_dir data/nyu_depth_v2/extracted \
    --output_dir data/yolo_nyu_dataset \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "Trained model: runs/nyu_anchor_detection/yolov8n_nyu/weights/best.pt"
echo ""
echo "To use in two-head model, update models/anchor_detection.py:"
echo "  model_path = 'runs/nyu_anchor_detection/yolov8n_nyu/weights/best.pt'"
echo ""
echo "Or run benchmark directly:"
echo "  python eval/nyu_distance/benchmark_all_models.py \\"
echo "    --models two-head-claude \\"
echo "    --device cuda"
echo ""
