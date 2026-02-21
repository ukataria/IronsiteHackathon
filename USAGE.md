# Usage Guide - Temporal Construction World Model

## Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **COLMAP**: 3D reconstruction software ([Installation Guide](https://colmap.github.io/install.html))
- **FFmpeg**: Video processing (for scene detection)
- **GPU**: Optional but recommended for faster processing

### Install COLMAP

**macOS (Homebrew):**
```bash
brew install colmap
```

**Ubuntu/Debian:**
```bash
sudo apt-get install colmap
```

**Windows:**
Download pre-built binaries from [COLMAP releases](https://github.com/colmap/colmap/releases)

### Verify Installation
```bash
colmap --version
```

---

## Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd IronsiteHackathon
```

2. **Install dependencies using uv:**
```bash
uv sync
```

This will install all required Python packages:
- opencv-python (video processing, frame extraction)
- numpy (numerical operations)
- open3d (point cloud processing, ICP alignment)
- scikit-learn (clustering, deduplication)
- scikit-image (SSIM for frame similarity)
- scipy (scientific computing)
- tqdm (progress bars)
- pyyaml (configuration)
- scenedetect (temporal segmentation)

---

## Quick Start

### Process a Construction Video

```bash
uv run python pipeline.py \
    --video data/raw/construction.mp4 \
    --video-id site_01
```

This runs the full pipeline:
1. Temporal segmentation (scene change detection)
2. Frame extraction (2-3 FPS)
3. Quality filtering (blur + dedup)
4. COLMAP 3D reconstruction
5. ICP alignment
6. Change detection
7. Timeline export

### Launch 3D Viewer

```bash
cd data/derived/site_01/viewer
python -m http.server 8000
```

Open http://localhost:8000 in your browser.

---

## Pipeline Configuration

Edit [config.yaml](config.yaml) to customize pipeline behavior:

### Temporal Segmentation
```yaml
segmentation:
  method: "scenedetect"  # or "manual"
  threshold: 27.0        # Lower = more sensitive (more segments)
  min_scene_len: 30      # Minimum frames per segment
```

### Frame Extraction
```yaml
extraction:
  fps: 2.5               # Frames per second (higher = more frames)
  max_frames: 800        # Total frame budget
  blur_threshold: 100.0  # Higher = only sharper frames
  similarity_threshold: 0.95  # SSIM threshold for dedup
```

### COLMAP Reconstruction
```yaml
colmap:
  feature_extractor: "sift"
  matcher: "exhaustive"  # or "sequential" for faster processing
  min_num_matches: 15
  camera_model: "SIMPLE_RADIAL"
  max_image_size: 1600   # Larger = better quality, slower
```

### Point Cloud Alignment
```yaml
alignment:
  method: "icp"
  reference: "segment_0"  # or "largest"
  max_iterations: 100
  threshold: 0.05        # ICP convergence threshold
  voxel_size: 0.05       # Downsampling voxel size
```

### Change Detection
```yaml
changes:
  nn_distance: 0.05      # Nearest neighbor threshold
  cluster_eps: 0.1       # DBSCAN epsilon
  min_cluster_size: 50   # Minimum points per cluster
```

---

## Advanced Usage

### Manual Temporal Segmentation

If automatic scene detection doesn't work well, provide manual time ranges:

```python
from pathlib import Path
from src.segmentation.segment_time import segment_video_manual

segments = segment_video_manual(
    video_path="data/raw/construction.mp4",
    segment_times=[
        (0, 30),      # Segment 0: 0-30 seconds
        (30, 60),     # Segment 1: 30-60 seconds
        (60, 120),    # Segment 2: 60-120 seconds
    ]
)
```

### Process Specific Segments Only

Skip completed stages:

```bash
uv run python pipeline.py \
    --video data/raw/construction.mp4 \
    --video-id site_01 \
    --skip-segmentation \
    --skip-extraction \
    --skip-reconstruction
```

This will only run alignment and change detection on existing reconstructions.

### Custom Reference Segment

By default, segment_0 is used as the reference frame. To use the largest segment instead:

Edit [config.yaml](config.yaml):
```yaml
alignment:
  reference: "largest"  # Use segment with most points
```

---

## Pipeline Output Structure

```
data/derived/{video_id}/
├── segments/
│   ├── segment_0/
│   │   ├── raw/                  # All extracted frames
│   │   ├── selected/             # Filtered high-quality frames
│   │   ├── colmap/               # COLMAP workspace
│   │   │   ├── database.db
│   │   │   ├── sparse/0/         # Sparse reconstruction
│   │   │   └── pointcloud.ply    # Raw point cloud
│   │   └── extraction_metadata.json
│   ├── segment_1/
│   └── segment_2/
├── aligned/
│   ├── reference.ply             # Reference point cloud
│   ├── segment_0_aligned.ply
│   ├── segment_1_aligned.ply
│   └── segment_2_aligned.ply
├── changes/
│   ├── segment_0_to_1/
│   │   ├── added.ply             # Green (new geometry)
│   │   ├── removed.ply           # Red (demolished)
│   │   └── changes_metadata.json
│   └── segment_1_to_2/
├── transforms/
│   ├── segment_1_aligned_transform.npy
│   └── segment_2_aligned_transform.npy
├── viewer/
│   ├── assets/                   # Point clouds for viewer
│   ├── timeline.json             # Segment timeline
│   └── metadata.json
└── pipeline_results.json         # Full pipeline metadata
```

---

## Viewer Controls

### 3D Viewer Interface

**Time Slider**: Switch between construction phases (segments)

**View Options**:
- Show Reference: Toggle baseline reference cloud
- Show Changes: Enable/disable change visualization
- Added Geometry: Green point clouds (new construction)
- Removed Geometry: Red point clouds (demolition)

**Camera Controls**:
- **Rotate**: Left mouse button + drag
- **Pan**: Right mouse button + drag
- **Zoom**: Mouse scroll wheel
- **Reset**: Click "Reset Camera" button

**Keyboard Shortcuts**:
- **←** Previous segment
- **→** Next segment

---

## Troubleshooting

### COLMAP Reconstruction Fails

**Symptom**: "Sparse reconstruction failed" or empty point cloud

**Solutions**:
1. **Increase frame overlap**: Edit `config.yaml`:
   ```yaml
   extraction:
     fps: 3.0  # Extract more frames
   ```

2. **Lower quality threshold**: Accept slightly blurry frames:
   ```yaml
   extraction:
     blur_threshold: 80.0  # Lower threshold
   ```

3. **Check frame quality**: Inspect `data/derived/{video_id}/segments/segment_0/selected/`
   - Need at least 20-30 frames
   - Frames should have good lighting and overlap

4. **Use sequential matcher** (faster, works for linear camera motion):
   ```yaml
   colmap:
     matcher: "sequential"
   ```

### Point Clouds Misaligned

**Symptom**: Segments don't overlap correctly in viewer

**Solutions**:
1. **Adjust ICP threshold**:
   ```yaml
   alignment:
     threshold: 0.1  # Increase for looser alignment
   ```

2. **Use manual reference**: Pick the best segment as reference
   ```yaml
   alignment:
     reference: "largest"
   ```

3. **Increase iterations**:
   ```yaml
   alignment:
     max_iterations: 200
   ```

### Change Detection Too Noisy

**Symptom**: Too many small change clusters

**Solutions**:
1. **Increase distance threshold**:
   ```yaml
   changes:
     nn_distance: 0.1  # More tolerant
   ```

2. **Larger clusters only**:
   ```yaml
   changes:
     min_cluster_size: 100  # Ignore small changes
   ```

3. **Adjust clustering**:
   ```yaml
   changes:
     cluster_eps: 0.2  # Merge nearby changes
   ```

### Viewer Shows "Timeline not found"

**Symptom**: Viewer fails to load timeline.json

**Solution**: Ensure you run the full pipeline to generate viewer assets, or manually export:

```python
from src.export.timeline import export_for_viewer

export_for_viewer(
    video_id="site_01",
    derived_dir="data/derived/site_01",
    viewer_output_dir="data/derived/site_01/viewer"
)
```

Then start the server from the viewer directory:
```bash
cd data/derived/site_01/viewer
python -m http.server 8000
```

---

## Performance Tips

### Speed Up Processing

1. **Lower frame count**:
   ```yaml
   extraction:
     fps: 2.0
     max_frames: 400
   ```

2. **Reduce image size**:
   ```yaml
   colmap:
     max_image_size: 1200
   ```

3. **Use sequential matcher**:
   ```yaml
   colmap:
     matcher: "sequential"
   ```

4. **Skip change detection** (if not needed):
   ```bash
   uv run python pipeline.py --video data/raw/video.mp4 --video-id site_01 --skip-changes
   ```

### Improve Quality

1. **More frames**:
   ```yaml
   extraction:
     fps: 3.0
     max_frames: 1200
   ```

2. **Higher image resolution**:
   ```yaml
   colmap:
     max_image_size: 2400
   ```

3. **Exhaustive matching**:
   ```yaml
   colmap:
     matcher: "exhaustive"
   ```

4. **Tighter alignment**:
   ```yaml
   alignment:
     threshold: 0.02
     voxel_size: 0.02
   ```

---

## Example Workflows

### Process 20-Second Construction Clip

```bash
# Full pipeline with default settings
uv run python pipeline.py \
    --video data/raw/construction_clip.mp4 \
    --video-id clip_01

# Launch viewer
cd data/derived/clip_01/viewer
python -m http.server 8000
# Open http://localhost:8000
```

### Process Multi-Hour Video (Fast Mode)

```bash
# Use manual segmentation to skip scene detection
# Create a script to segment by time ranges
python -c "
from src.segmentation.segment_time import segment_video_manual
from src.utils import save_json

segments = segment_video_manual(
    'data/raw/long_video.mp4',
    [(0, 300), (300, 600), (600, 900)]  # Every 5 minutes
)

save_json({'segments': segments}, 'data/derived/long_video/segments_metadata/segments.json')
"

# Run pipeline skipping segmentation
uv run python pipeline.py \
    --video data/raw/long_video.mp4 \
    --video-id long_video \
    --skip-segmentation
```

### Regenerate Alignment Only

If you need to re-align with different parameters:

```bash
# Modify config.yaml first, then:
python -c "
from src.alignment.align import align_all_segments
from pathlib import Path

segment_clouds = sorted(Path('data/derived/site_01/segments').glob('segment_*/colmap/pointcloud.ply'))

align_all_segments(
    segment_clouds=segment_clouds,
    reference_index=0,
    output_dir='data/derived/site_01/aligned'
)
"
```

---

## API Reference

See individual module docstrings for detailed API documentation:

- [src/segmentation/segment_time.py](src/segmentation/segment_time.py) - Temporal segmentation
- [src/extraction/extract.py](src/extraction/extract.py) - Frame extraction and filtering
- [src/reconstruction/colmap_wrapper.py](src/reconstruction/colmap_wrapper.py) - COLMAP wrapper
- [src/alignment/align.py](src/alignment/align.py) - ICP point cloud alignment
- [src/changes/detect.py](src/changes/detect.py) - Change detection
- [src/export/timeline.py](src/export/timeline.py) - Timeline export

---

## Support

For issues, questions, or contributions, please open an issue on GitHub.
