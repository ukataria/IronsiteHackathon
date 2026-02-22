#!/bin/bash
#
# Full Benchmark Runner for GPU Server
#
# This script runs the complete VLM benchmark including:
# - All API models (GPT-4o, Claude, Gemini)
# - All open-source models (InternVL3, Qwen2.5-VL, LLaVA, etc.)
# - Two-head spatial anchor calibration model
#
# Requirements:
# - CUDA-capable GPU (16GB+ VRAM recommended)
# - Python 3.9+
# - API keys in .env file

set -e

echo "=========================================="
echo "FULL VLM BENCHMARK - GPU MODE"
echo "=========================================="
echo ""

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. CUDA may not be available."
    echo "   Some models require GPU. Proceeding anyway..."
else
    echo "✓ CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Check for required packages
echo "Checking dependencies..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || {
    echo "✗ PyTorch not found. Installing GPU requirements..."
    pip install -r requirements-gpu.txt
}

python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" 2>/dev/null || {
    echo "✗ Transformers not found. Installing..."
    pip install transformers accelerate
}

python -c "import ultralytics; print(f'✓ Ultralytics (YOLO) installed')" 2>/dev/null || {
    echo "✗ YOLO not found. Installing..."
    pip install ultralytics
}

echo ""
echo "✓ Dependencies ready"
echo ""

# Check API keys
echo "Checking API keys..."
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. API models will be skipped."
else
    source .env
    [ -n "$OPENAI_API_KEY" ] && echo "✓ OPENAI_API_KEY set"
    [ -n "$ANTHROPIC_API_KEY" ] && echo "✓ ANTHROPIC_API_KEY set"
    [ -n "$GEMINI_API_KEY" ] && echo "✓ GEMINI_API_KEY set"
fi
echo ""

# Configuration
NUM_IMAGES=${NUM_IMAGES:-10}
PAIRS_PER_IMAGE=${PAIRS_PER_IMAGE:-3}
TOTAL_QUERIES=$((NUM_IMAGES * PAIRS_PER_IMAGE))

echo "=========================================="
echo "BENCHMARK CONFIGURATION"
echo "=========================================="
echo "Dataset: NYU Depth V2"
echo "Test images: $NUM_IMAGES"
echo "Pairs per image: $PAIRS_PER_IMAGE"
echo "Total queries: $TOTAL_QUERIES"
echo "Device: CUDA (GPU)"
echo ""

# Ask user which models to test
echo "Select models to test:"
echo "  1) All models (API + Open Source + Two-Head)"
echo "  2) API models only (GPT-4o, Claude, Gemini)"
echo "  3) Two-Head model only (our approach)"
echo "  4) Custom selection"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        MODELS=""  # Empty = all models
        echo "Testing all models..."
        ;;
    2)
        MODELS="--models gpt-4o claude-sonnet-4 gemini-2.5-pro"
        echo "Testing API models only..."
        ;;
    3)
        MODELS="--models two-head-claude"
        echo "Testing Two-Head model only..."
        ;;
    4)
        echo "Available models:"
        echo "  API: gpt-4o, claude-sonnet-4, gemini-2.5-pro"
        echo "  Open: internvl3, qwen2.5-vl, llava-onevision"
        echo "  Two-Head: two-head-claude"
        read -p "Enter model names (space-separated): " custom_models
        MODELS="--models $custom_models"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "STARTING BENCHMARK"
echo "=========================================="
echo ""

# Run benchmark
python eval/nyu_distance/benchmark_all_models.py \
    --nyu_data_dir data/nyu_depth_v2/extracted \
    --num_images $NUM_IMAGES \
    --pairs_per_image $PAIRS_PER_IMAGE \
    --out_dir outputs/nyu_benchmark_gpu \
    --device cuda \
    $MODELS

echo ""
echo "=========================================="
echo "BENCHMARK COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: outputs/nyu_benchmark_gpu/"
echo "  - benchmark_results.json (full results)"
echo "  - benchmark_comparison.png (visualization)"
echo "  - marked_images/ (test images with annotations)"
echo ""
echo "To view results:"
echo "  cat outputs/nyu_benchmark_gpu/benchmark_results.json | jq"
echo "  open outputs/nyu_benchmark_gpu/benchmark_comparison.png"
echo ""
