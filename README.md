# Temporal Construction World Model

**4D reconstruction pipeline for construction site evolution tracking**

Transform construction walkthrough videos into interactive 3D temporal visualizations with aligned point clouds and change detection.

---

## What This Does

Converts construction video → Temporal 3D point cloud sequence with:
- **Segmented time chunks** (construction phases)
- **High-quality frames** (400-800 selected via blur filtering)
- **3D point clouds** per segment (COLMAP reconstruction)
- **Aligned coordinate frame** (ICP registration)
- **Change detection** (added/removed geometry)
- **Interactive 3D viewer** with time slider

---

## Quick Start

```bash
# Install dependencies
uv sync

# Process a construction video
uv run python pipeline.py --video data/raw/construction.mp4 --video-id site_01

# Launch viewer
cd app
python -m http.server 8000
# Open http://localhost:8000
```

---

## Pipeline Stages

```
Video Input
    ↓
[1] Temporal Segmentation (scene change detection)
    ↓
[2] Frame Extraction (2-3 FPS)
    ↓
[3] Quality Filtering (blur detection, deduplication)
    ↓
[4] Frame Selection (400-800 best frames)
    ↓
[5] COLMAP Reconstruction (per segment)
    ↓
[6] Point Cloud Export (.ply)
    ↓
[7] Temporal Alignment (ICP → reference frame)
    ↓
[8] Change Detection (geometry diff)
    ↓
[9] Timeline Export (timeline.json + assets)
    ↓
3D Viewer (slider-based visualization)
```

---

## Output Structure

```
data/derived/{video_id}/
├── segments/
│   ├── segment_0/
│   │   ├── frames/           # Selected frames
│   │   ├── colmap/           # COLMAP workspace
│   │   └── pointcloud.ply    # Raw reconstruction
│   ├── segment_1/
│   └── segment_2/
├── aligned/
│   ├── segment_0_aligned.ply
│   ├── segment_1_aligned.ply
│   ├── segment_2_aligned.ply
│   └── reference.ply
├── changes/
│   ├── added_1.ply           # Green (new geometry)
│   ├── removed_1.ply         # Red (demolished)
│   └── clusters_1.json
├── transforms/
│   ├── segment_1_to_ref.json
│   └── timeline.json
└── metadata.json
```

---

## Interactive Viewer

The web viewer provides:
- **3D point cloud rendering** (Three.js)
- **Time slider** (discrete segment switching)
- **Change overlays** (toggle added/removed geometry)
- **Camera controls** (orbit, pan, zoom)
- **No backend required** (static HTML/JS)

---

## Dependencies

**Core:**
- Python 3.11+
- COLMAP (3D reconstruction)
- FFmpeg (video processing)

**Python packages:**
- opencv-python (frame extraction, blur detection)
- numpy (point cloud processing)
- open3d (ICP alignment, change detection)
- scikit-learn (clustering, deduplication)

**Viewer:**
- Three.js (3D rendering)
- PLYLoader (point cloud loading)

---

## Configuration

Edit `config.yaml`:

```yaml
segmentation:
  method: "scenedetect"  # or "manual"
  threshold: 27.0        # Scene change sensitivity

extraction:
  fps: 2.5               # Frames per second
  max_frames: 800        # Total frame budget
  blur_threshold: 100.0  # Laplacian variance cutoff

colmap:
  feature_extractor: "sift"
  matcher: "exhaustive"
  min_num_matches: 15

alignment:
  method: "icp"
  reference: "segment_0"  # or "largest" or "manual"
  max_iterations: 100
  threshold: 0.05

changes:
  nn_distance: 0.05      # Nearest neighbor threshold
  cluster_eps: 0.1       # DBSCAN epsilon
  min_cluster_size: 50   # Minimum points per change
```

---

## Usage Examples

### Process entire video
```bash
uv run python pipeline.py \
    --video data/raw/construction_site_01.mp4 \
    --video-id site_01 \
    --config config.yaml
```

### Process specific segments
```bash
uv run python pipeline.py \
    --video data/raw/construction_site_01.mp4 \
    --video-id site_01 \
    --segments 0,1,2 \
    --skip-alignment
```

### Regenerate alignment only
```bash
uv run python src/alignment/align.py \
    --base-dir data/derived/site_01/segments \
    --reference segment_0 \
    --output data/derived/site_01/aligned
```

### Detect changes between segments
```bash
uv run python src/changes/detect.py \
    --cloud-a data/derived/site_01/aligned/segment_0_aligned.ply \
    --cloud-b data/derived/site_01/aligned/segment_1_aligned.ply \
    --output data/derived/site_01/changes
```

---

## Project Structure

```
├── data/
│   ├── raw/              # Input videos
│   └── derived/          # Pipeline outputs
├── src/
│   ├── segmentation/     # Temporal segmentation
│   ├── extraction/       # Frame extraction + filtering
│   ├── reconstruction/   # COLMAP wrapper
│   ├── alignment/        # Point cloud ICP
│   ├── changes/          # Change detection
│   └── export/           # Timeline + asset packaging
├── app/                  # 3D viewer (HTML/JS)
│   ├── index.html
│   ├── main.js
│   └── assets/           # Symlink to derived data
├── pipeline.py           # Main orchestration script
├── config.yaml           # Pipeline configuration
└── pyproject.toml        # Dependencies
```

---

## Future Extensions

- **Semantic labeling**: Wall, ladder, worker detection
- **2D-to-3D projection**: Map segmentation masks to point clouds
- **Persistent tracking**: Track objects across time
- **Multi-session comparison**: Compare different construction sites
- **NeRF/3DGS rendering**: Neural radiance fields for photo-realistic views

---

## Troubleshooting

**COLMAP reconstruction fails:**
- Ensure sufficient frame overlap (increase FPS)
- Check lighting consistency
- Verify feature matches (`colmap matches_importer --verify_matches`)

**Point clouds misaligned:**
- Adjust ICP threshold in config
- Manually specify reference segment
- Increase frame quality (lower blur threshold)

**Change detection too noisy:**
- Increase `nn_distance` threshold
- Adjust `cluster_eps` and `min_cluster_size`
- Filter by statistical outlier removal

---

## License

MIT

---

## Acknowledgments

Built for the Ironsite Spatial Intelligence Hackathon
