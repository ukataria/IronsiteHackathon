"""
4D Construction Site Reconstruction Pipeline.

Takes a single video, splits it into time periods, reconstructs each period
with COLMAP + 3D Gaussian Splatting, aligns periods via ICP, and outputs
an interactive HTML viewer with a time slider.

Graceful fallback at every stage: COLMAP/gsplat failure → depth-based point clouds.

Setup:
    uv add gsplat opencv-python torch torchvision transformers pillow numpy open3d plotly
    sudo apt-get install -y colmap   # on Linux/Vast.ai

Usage:
    # Full pipeline
    python pipeline_4d.py --video videoplayback.mp4 --periods 2

    # Fast test (depth-only, no COLMAP/gsplat/SAM2)
    python pipeline_4d.py --video videoplayback.mp4 --periods 2 --no-colmap --no-splat --no-sam2

    # Headless (Vast.ai)
    python pipeline_4d.py --video videoplayback.mp4 --periods 4 --no-viewer
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
PERIODS_DIR = DATA_DIR / "periods"
OUTPUT_DIR = DATA_DIR / "output"

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
DETECTOR_MODEL_ID = "google/owlv2-base-patch16-ensemble"

CAMERA_FOV_DEG = 69.0
MAX_DEPTH_SCALE = 8.0
SUBSAMPLE = 4

TARGET_FPS = 2.5
BLUR_THRESHOLD = 100.0
VOXEL_SIZE = 0.05
ICP_MAX_ITER = 50
GSPLAT_ITERATIONS = 7000

DYNAMIC_OBJECT_LABELS = [
    "person", "construction worker", "worker",
    "machinery", "forklift", "crane", "vehicle",
]


# ── Setup ──────────────────────────────────────────────────────────────────────

def setup_dirs(n_periods: int) -> list[dict[str, Path]]:
    """Create output directory tree; return list of per-period path dicts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    period_paths = []
    for i in range(n_periods):
        base = PERIODS_DIR / f"period_{i:02d}"
        dirs = {
            "base":   base,
            "frames": base / "frames",
            "masked": base / "masked",
            "depth":  base / "depth",
            "colmap": base / "colmap",
            "splat":  base / "splat",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        period_paths.append(dirs)
    return period_paths


def get_device() -> str:
    """Return the best available compute device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    print("  [warn] No GPU found — processing will be slow on CPU")
    return "cpu"


# ── Stage 1: Video Split + Frame Extraction ────────────────────────────────────

def split_video_into_periods(
    video_path: Path,
    n_periods: int,
    period_dirs: list[dict[str, Path]],
    target_fps: float = TARGET_FPS,
    blur_threshold: float = BLUR_THRESHOLD,
) -> list[list[Path]]:
    """
    Split video into N equal chunks; extract sharp frames at target_fps per chunk.
    Returns period_frames[i] = list of saved frame paths for period i.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total} frames @ {src_fps:.1f} fps")

    boundaries = np.linspace(0, total, n_periods + 1, dtype=int)
    period_frame_paths: list[list[Path]] = []

    for i in range(n_periods):
        frames_dir = period_dirs[i]["frames"]
        existing = sorted(frames_dir.glob("*.png"))
        if existing:
            print(f"  [cache] period_{i:02d}: {len(existing)} frames")
            period_frame_paths.append(existing)
            continue

        start, end = int(boundaries[i]), int(boundaries[i + 1])
        chunk_len = end - start
        n_target = max(1, int(chunk_len / src_fps * target_fps))
        indices = np.linspace(start, end - 1, n_target, dtype=int)

        saved: list[Path] = []
        blurry = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score < blur_threshold:
                blurry += 1
                continue
            out_path = frames_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)

        print(
            f"  period_{i:02d}: {len(saved)} sharp frames "
            f"({blurry} blurry frames discarded)"
        )
        period_frame_paths.append(saved)

    cap.release()
    return period_frame_paths


# ── Stage 2: SAM2 Dynamic Object Masking ──────────────────────────────────────

def load_sam2_pipeline(device: str) -> tuple:
    """
    Load OWLv2 (text→boxes) and SAM2 (boxes→masks) from HuggingFace transformers.
    Returns (sam2_model, sam2_processor, detector_pipe).
    """
    from transformers import pipeline as hf_pipeline, Sam2Model, Sam2Processor

    print(f"  Loading OWLv2 detector: {DETECTOR_MODEL_ID}")
    detector_pipe = hf_pipeline(
        "zero-shot-object-detection",
        model=DETECTOR_MODEL_ID,
        device=0 if device == "cuda" else -1,
    )

    print(f"  Loading SAM2: {SAM2_MODEL_ID}")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    sam2_model = Sam2Model.from_pretrained(
        SAM2_MODEL_ID, torch_dtype=torch_dtype
    ).to(device)
    sam2_processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)

    return sam2_model, sam2_processor, detector_pipe


def mask_frame(
    frame_path: Path,
    masked_path: Path,
    sam2_model,
    sam2_processor,
    detector_pipe,
    device: str,
    score_threshold: float = 0.15,
) -> Path:
    """
    Detect dynamic objects with OWLv2, then mask precisely with SAM2.
    Fills masked pixels with black. Caches result.
    Returns masked_path on success, frame_path on failure/no detections.
    """
    if masked_path.exists():
        return masked_path

    try:
        image = Image.open(frame_path).convert("RGB")
        detections = detector_pipe(
            image,
            candidate_labels=DYNAMIC_OBJECT_LABELS,
            threshold=score_threshold,
        )
        if not detections:
            image.save(str(masked_path))
            return masked_path

        input_boxes = [
            [[d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]]]
            for d in detections
        ]
        inputs = sam2_processor(
            images=image,
            input_boxes=[input_boxes],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = sam2_model(**inputs, multimask_output=False)

        masks = sam2_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        combined = masks.squeeze(1).any(dim=0).numpy()

        frame_arr = np.array(image)
        frame_arr[combined] = 0
        Image.fromarray(frame_arr).save(str(masked_path))
        return masked_path

    except Exception as e:
        print(f"    [warn] masking failed for {frame_path.name}: {e}")
        return frame_path


def mask_period_frames(
    frame_paths: list[Path],
    masked_dir: Path,
    sam2_model,
    sam2_processor,
    detector_pipe,
    device: str,
) -> list[Path]:
    """Apply SAM2 masking to all frames in a period. Returns masked frame paths."""
    masked_paths = []
    for fp in frame_paths:
        out = masked_dir / fp.name
        result = mask_frame(fp, out, sam2_model, sam2_processor, detector_pipe, device)
        masked_paths.append(result)
    return masked_paths


# ── Stage 3: Depth Estimation (Fallback Point Clouds) ─────────────────────────

def load_depth_pipeline(device: str):
    """Load Depth Anything V2 via HuggingFace transformers pipeline."""
    from transformers import pipeline as hf_pipeline

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL_ID,
        device=0 if device == "cuda" else -1,
        torch_dtype=torch_dtype,
    )
    print(f"  Depth model loaded: {DEPTH_MODEL_ID}")
    return pipe


def estimate_depth(pipe, frame_path: Path, depth_dir: Path) -> np.ndarray:
    """
    Run depth estimation on one frame, cache to depth_dir.
    Returns float32 depth array (H, W) in [0, MAX_DEPTH_SCALE].
    """
    npy_path = depth_dir / f"{frame_path.stem}.npy"
    png_path = depth_dir / f"{frame_path.stem}.png"

    if npy_path.exists():
        return np.load(str(npy_path))

    image = Image.open(frame_path).convert("RGB")
    result = pipe(image)
    raw: torch.Tensor = result["predicted_depth"].squeeze()
    depth_f32 = raw.float().numpy()

    d_min, d_max = float(depth_f32.min()), float(depth_f32.max())
    depth_scaled = (depth_f32 - d_min) / (d_max - d_min + 1e-8) * MAX_DEPTH_SCALE

    np.save(str(npy_path), depth_scaled)
    vis_u8 = (depth_scaled / MAX_DEPTH_SCALE * 255).astype(np.uint8)
    cv2.imwrite(str(png_path), vis_u8)
    return depth_scaled


def estimate_depth_for_period(
    frame_paths: list[Path],
    depth_dir: Path,
    depth_pipe,
) -> dict[Path, np.ndarray]:
    """Run depth estimation for all frames in a period. Returns {frame_path: depth}."""
    depth_maps: dict[Path, np.ndarray] = {}
    for fp in frame_paths:
        depth_maps[fp] = estimate_depth(depth_pipe, fp, depth_dir)
    return depth_maps


def pinhole_intrinsics(
    width: int, height: int, fov_deg: float
) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for an estimated pinhole camera."""
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy


def build_point_cloud(
    frame_path: Path,
    depth: np.ndarray,
    fov_deg: float = CAMERA_FOV_DEG,
    subsample: int = SUBSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project image pixels to 3D using depth map.
    Returns (points (N,3), colors (N,3)) float32.
    """
    bgr = cv2.imread(str(frame_path))
    h_img, w_img = bgr.shape[:2]
    if depth.shape != (h_img, w_img):
        depth = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    fx, fy, cx, cy = pinhole_intrinsics(w_img, h_img, fov_deg)
    ys, xs = np.mgrid[0:h_img:subsample, 0:w_img:subsample]
    ys, xs = ys.flatten(), xs.flatten()
    z = depth[ys, xs]

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    colors = bgr[ys, xs][:, ::-1].astype(np.float32) / 255.0

    valid = (z > 0.05) & (z < MAX_DEPTH_SCALE * 0.98)
    return points[valid], colors[valid]


def build_period_point_cloud(
    frame_paths: list[Path],
    depth_maps: dict[Path, np.ndarray],
    fov_deg: float,
    subsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge point clouds from all frames in a period into one aggregate cloud."""
    all_pts, all_cols = [], []
    for fp in frame_paths:
        pts, cols = build_point_cloud(fp, depth_maps[fp], fov_deg, subsample)
        all_pts.append(pts)
        all_cols.append(cols)
    return np.concatenate(all_pts), np.concatenate(all_cols)


def save_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    """Write a colored point cloud to a PLY file via Open3D."""
    try:
        import open3d as o3d
    except ImportError:
        print("  [warn] open3d not installed — skipping PLY save")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"  [saved] {out_path.name} ({len(points):,} points)")


# ── Stage 4: COLMAP Structure from Motion ─────────────────────────────────────

def run_colmap(
    frame_dir: Path,
    colmap_dir: Path,
    use_gpu: bool = True,
) -> bool:
    """
    Run COLMAP feature extraction → matching → mapper via subprocess.
    Returns True on success, False if COLMAP not installed or reconstruction fails.
    """
    cache_signal = colmap_dir / "sparse" / "0" / "cameras.bin"
    if cache_signal.exists():
        print(f"  [cache] COLMAP sparse reconstruction already exists")
        return True

    result = subprocess.run(["colmap", "--version"], capture_output=True)
    if result.returncode != 0:
        print("  [warn] COLMAP not found — falling back to depth-based point clouds")
        print("         Install with: sudo apt-get install -y colmap")
        return False

    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    gpu_flag = "1" if use_gpu else "0"

    try:
        print("  Running COLMAP feature extraction ...")
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(frame_dir),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--ImageReader.camera_mode", "1",
            "--SiftExtraction.use_gpu", gpu_flag,
        ], check=True)

        print("  Running COLMAP exhaustive matching ...")
        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", gpu_flag,
        ], check=True)

        print("  Running COLMAP mapper ...")
        subprocess.run([
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(frame_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.min_num_matches", "15",
        ], check=True)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [warn] COLMAP failed: {e} — falling back to depth-based point clouds")
        return False

    if not cache_signal.exists():
        print("  [warn] COLMAP ran but produced no reconstruction (too few matching frames)")
        return False

    print("  COLMAP reconstruction complete")
    return True


# ── Stage 5: 3D Gaussian Splatting ────────────────────────────────────────────

def _find_gsplat_trainer() -> Optional[Path]:
    """Locate gsplat's simple_trainer.py after installation."""
    try:
        import gsplat
        gsplat_pkg_dir = Path(gsplat.__file__).parent
        candidates = [
            gsplat_pkg_dir.parent / "examples" / "simple_trainer.py",
            gsplat_pkg_dir / "examples" / "simple_trainer.py",
            Path(sys.prefix) / "examples" / "simple_trainer.py",
        ]
        for c in candidates:
            if c.exists():
                return c
    except ImportError:
        pass
    return None


def run_gsplat(
    frame_dir: Path,
    colmap_dir: Path,
    splat_dir: Path,
    iterations: int = GSPLAT_ITERATIONS,
    device: str = "cuda",
) -> bool:
    """
    Train 3D Gaussian Splatting via gsplat's simple_trainer.py.
    Expects COLMAP output in colmap_dir/sparse/0/.
    Returns True on success, False on failure.
    """
    ckpts = list((splat_dir / "ckpts").glob("*.pt")) if (splat_dir / "ckpts").exists() else []
    if ckpts:
        print(f"  [cache] gsplat checkpoint already exists")
        return True

    try:
        import gsplat  # noqa: F401
    except ImportError:
        print("  [warn] gsplat not installed — run: uv add gsplat")
        return False

    trainer = _find_gsplat_trainer()
    if trainer is None:
        print("  [warn] Could not locate gsplat simple_trainer.py")
        return False

    # gsplat expects images/ and sparse/0/ under data_dir
    images_link = colmap_dir / "images"
    if not images_link.exists():
        try:
            images_link.symlink_to(frame_dir.resolve())
        except OSError:
            # Windows fallback: copy instead of symlink
            import shutil
            shutil.copytree(str(frame_dir), str(images_link))

    print(f"  Training Gaussian Splat ({iterations} iterations) ...")
    try:
        subprocess.run([
            sys.executable, str(trainer), "default",
            "--data_dir", str(colmap_dir),
            "--result_dir", str(splat_dir),
            "--max_steps", str(iterations),
            "--disable_viewer",
        ], check=True, timeout=7200)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [warn] gsplat training failed: {e}")
        return False

    ckpts = list((splat_dir / "ckpts").glob("*.pt"))
    if not ckpts:
        print("  [warn] gsplat ran but produced no checkpoint")
        return False

    print("  Gaussian Splatting training complete")
    return True


def extract_splat_point_cloud(
    splat_dir: Path,
    max_points: int = 200_000,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Extract (points, colors) from a trained gsplat checkpoint.
    Returns None if extraction fails.
    """
    ckpts = sorted((splat_dir / "ckpts").glob("*.pt"))
    if not ckpts:
        return None

    ckpt = torch.load(str(ckpts[-1]), map_location="cpu")
    state = ckpt.get("splats", ckpt)  # handle both wrapped and flat checkpoints

    # Try multiple candidate key names across gsplat versions
    means = None
    for key in ("means", "splats.means", "params.means"):
        if key in state:
            means = state[key].float().numpy()
            break
    if means is None:
        print(f"  [warn] Could not find 'means' in checkpoint. Available keys: {list(state.keys())[:10]}")
        return None

    colors = None
    for key in ("sh0", "splats.sh0", "params.sh0", "features_dc"):
        if key in state:
            raw = state[key].float()
            # SH degree-0: shape (N, 1, 3) or (N, 3)
            if raw.dim() == 3:
                raw = raw.squeeze(1)
            colors = (raw * 0.28209479 + 0.5).clamp(0, 1).numpy()
            break
    if colors is None:
        print(f"  [warn] Could not find color SH coefficients. Available keys: {list(state.keys())[:10]}")
        return None

    if len(means) > max_points:
        idx = np.random.choice(len(means), max_points, replace=False)
        means, colors = means[idx], colors[idx]

    return means.astype(np.float32), colors.astype(np.float32)


# ── Stage 6: ICP Alignment ─────────────────────────────────────────────────────

def _preprocess_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
):
    """Downsample, estimate normals, compute FPFH features. Returns (pcd_down, fpfh)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return pcd_down, fpfh


def _register_pair(
    source_down,
    source_fpfh,
    target_down,
    target_fpfh,
    voxel_size: float,
) -> np.ndarray:
    """RANSAC global alignment + Point-to-Plane ICP. Returns 4x4 transform."""
    import open3d as o3d

    dist_threshold = voxel_size * 1.5
    result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
    )

    icp_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        max_correspondence_distance=icp_threshold,
        init=result_global.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER),
    )
    return result_icp.transformation


def align_point_clouds(
    period_clouds: list[tuple[np.ndarray, np.ndarray]],
    voxel_size: float = VOXEL_SIZE,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """
    Align all period point clouds to period 0 using FPFH + ICP.
    Returns (aligned_clouds, transforms).
    """
    try:
        import open3d as o3d  # noqa: F401
    except ImportError:
        print("  [warn] open3d not installed — skipping ICP alignment")
        return period_clouds, [np.eye(4)] * len(period_clouds)

    aligned: list[tuple[np.ndarray, np.ndarray]] = [period_clouds[0]]
    transforms: list[np.ndarray] = [np.eye(4)]

    target_down, target_fpfh = _preprocess_cloud(*period_clouds[0], voxel_size)

    for i, (pts, cols) in enumerate(period_clouds[1:], start=1):
        print(f"  Aligning period_{i:02d} to period_00 ...")
        try:
            source_down, source_fpfh = _preprocess_cloud(pts, cols, voxel_size)
            T = _register_pair(source_down, source_fpfh, target_down, target_fpfh, voxel_size)

            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            pcd.transform(T)

            aligned_pts = np.asarray(pcd.points).astype(np.float32)
            aligned_cols = np.asarray(pcd.colors).astype(np.float32)
            aligned.append((aligned_pts, aligned_cols))
            transforms.append(T)

            np.save(str(OUTPUT_DIR / f"transform_{i:02d}.npy"), T)
            print(f"  period_{i:02d} aligned (fitness: {_register_pair.__doc__})")
        except Exception as e:
            print(f"  [warn] ICP failed for period_{i:02d}: {e} — using unaligned cloud")
            aligned.append((pts, cols))
            transforms.append(np.eye(4))

    return aligned, transforms


# ── Stage 7: 4D HTML Viewer ────────────────────────────────────────────────────

def _compute_global_bounds(
    period_clouds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[float, float]]:
    """Compute bounding box across all periods for fixed axis ranges."""
    all_pts = np.concatenate([p for p, _ in period_clouds])
    return {
        "x": (float(all_pts[:, 0].min()), float(all_pts[:, 0].max())),
        "y": (float(all_pts[:, 2].min()), float(all_pts[:, 2].max())),
        "z": (float(-all_pts[:, 1].max()), float(-all_pts[:, 1].min())),
    }


def export_4d_html(
    period_clouds: list[tuple[np.ndarray, np.ndarray]],
    period_labels: list[str],
    out_path: Path,
    max_pts_per_period: int = 50_000,
) -> None:
    """
    Export an interactive Plotly HTML with time slider and Play/Pause buttons.
    Each slider step = one time period. Fixed camera/axis ranges between frames.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [warn] plotly not installed — skipping HTML export")
        return

    bounds = _compute_global_bounds(period_clouds)

    def make_trace(pts: np.ndarray, cols: np.ndarray, label: str) -> go.Scatter3d:
        step = max(1, len(pts) // max_pts_per_period)
        p, c = pts[::step], cols[::step]
        hex_colors = [
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            for r, g, b in c
        ]
        return go.Scatter3d(
            x=p[:, 0], y=p[:, 2], z=-p[:, 1],
            mode="markers",
            marker=dict(size=1.2, color=hex_colors, opacity=0.85),
            name=label,
        )

    traces = [make_trace(pts, cols, label) for (pts, cols), label in zip(period_clouds, period_labels)]

    # Use restyle (toggle visibility) rather than animate to avoid camera reset
    slider_steps = []
    for i, label in enumerate(period_labels):
        visible = [j == i for j in range(len(period_labels))]
        slider_steps.append(dict(
            label=label,
            method="restyle",
            args=[{"visible": visible}],
        ))

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title="Ghost Blueprint — 4D Construction Site",
            scene=dict(
                xaxis=dict(title="X", range=list(bounds["x"])),
                yaxis=dict(title="Depth (Z)", range=list(bounds["y"])),
                zaxis=dict(title="Up (Y)", range=list(bounds["z"])),
                aspectmode="data",
                bgcolor="rgb(10,10,20)",
            ),
            paper_bgcolor="rgb(10,10,20)",
            font_color="white",
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Period: ", font=dict(color="white")),
                steps=slider_steps,
                pad=dict(t=10),
                bgcolor="rgba(50,50,50,0.8)",
                font=dict(color="white"),
            )],
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            margin=dict(l=0, r=0, t=50, b=80),
        ),
    )

    # Show only first period by default
    for i, trace in enumerate(fig.data):
        trace.visible = (i == 0)

    fig.write_html(str(out_path))
    print(f"  [saved] {out_path}")


# ── Argument Parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4D construction site reconstruction from a single video"
    )
    p.add_argument("--video", type=Path, required=True,
                   help="Path to input video")
    p.add_argument("--periods", type=int, default=4,
                   help="Number of time periods (default: 4)")
    p.add_argument("--fov", type=float, default=CAMERA_FOV_DEG,
                   help="Camera horizontal FOV in degrees (default: 69)")
    p.add_argument("--subsample", type=int, default=SUBSAMPLE,
                   help="Pixel subsampling factor for point cloud (default: 4)")
    p.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD,
                   help="Laplacian variance below this = blurry (default: 100)")
    p.add_argument("--iterations", type=int, default=GSPLAT_ITERATIONS,
                   help="Gaussian splatting training steps (default: 7000)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Human-readable labels for each period")
    p.add_argument("--no-colmap", action="store_true",
                   help="Skip COLMAP, use depth-based point clouds only")
    p.add_argument("--no-splat", action="store_true",
                   help="Skip Gaussian splatting, use depth-based point clouds only")
    p.add_argument("--no-sam2", action="store_true",
                   help="Skip SAM2 dynamic object masking")
    p.add_argument("--no-viewer", action="store_true",
                   help="Skip interactive viewer (headless / Vast.ai)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.video.exists():
        print(f"Error: video not found at '{args.video}'")
        sys.exit(1)

    labels = args.labels or [f"Period {i+1}" for i in range(args.periods)]
    labels = (labels * args.periods)[: args.periods]

    period_dirs = setup_dirs(args.periods)
    device = get_device()
    print(f"Device: {device}\n")

    # ── [1/6] Split video + extract sharp frames ──────────────────────────────
    print(f"[1/6] Splitting '{args.video.name}' into {args.periods} periods ...")
    period_frame_paths = split_video_into_periods(
        args.video, args.periods, period_dirs,
        target_fps=TARGET_FPS,
        blur_threshold=args.blur_threshold,
    )
    print()

    # ── [2/6] SAM2 dynamic object masking ────────────────────────────────────
    if not args.no_sam2:
        print("[2/6] Loading SAM2 + OWLv2 for dynamic object masking ...")
        try:
            sam2_model, sam2_processor, detector_pipe = load_sam2_pipeline(device)
            for i, frame_paths in enumerate(period_frame_paths):
                print(f"  Masking period_{i:02d} ({len(frame_paths)} frames) ...")
                period_frame_paths[i] = mask_period_frames(
                    frame_paths, period_dirs[i]["masked"],
                    sam2_model, sam2_processor, detector_pipe, device,
                )
        except Exception as e:
            print(f"  [warn] SAM2 setup failed ({e}) — using unmasked frames")
    else:
        print("[2/6] Skipping SAM2 masking (--no-sam2)")
    print()

    # ── [3/6] Depth estimation (always runs — baseline fallback clouds) ───────
    print("[3/6] Loading depth model and estimating depth ...")
    depth_pipe = load_depth_pipeline(device)
    period_depth_clouds: list[tuple[np.ndarray, np.ndarray]] = []

    for i, frame_paths in enumerate(period_frame_paths):
        if not frame_paths:
            print(f"  [warn] period_{i:02d} has no frames — skipping")
            period_depth_clouds.append((np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)))
            continue
        print(f"  period_{i:02d}: estimating depth for {len(frame_paths)} frames ...")
        depth_maps = estimate_depth_for_period(frame_paths, period_dirs[i]["depth"], depth_pipe)
        pts, cols = build_period_point_cloud(frame_paths, depth_maps, args.fov, args.subsample)
        save_ply(pts, cols, period_dirs[i]["base"] / "point_cloud_depth.ply")
        period_depth_clouds.append((pts, cols))
    print()

    # ── [4/6] COLMAP + gsplat per period ─────────────────────────────────────
    print("[4/6] Running COLMAP + Gaussian Splatting per period ...")
    period_final_clouds: list[tuple[np.ndarray, np.ndarray]] = []

    for i, frame_paths in enumerate(period_frame_paths):
        print(f"\n  === period_{i:02d} ({labels[i]}) ===")
        used_splat = False

        # Pick the best available frame directory (masked > frames)
        masked_dir = period_dirs[i]["masked"]
        has_masked = masked_dir.exists() and any(masked_dir.iterdir())
        active_frame_dir = masked_dir if has_masked else period_dirs[i]["frames"]

        if not args.no_colmap and frame_paths:
            colmap_ok = run_colmap(active_frame_dir, period_dirs[i]["colmap"])
        else:
            colmap_ok = False
            if args.no_colmap:
                print("  Skipping COLMAP (--no-colmap)")

        if colmap_ok and not args.no_splat:
            splat_ok = run_gsplat(
                active_frame_dir, period_dirs[i]["colmap"],
                period_dirs[i]["splat"], args.iterations, device,
            )
            if splat_ok:
                splat_cloud = extract_splat_point_cloud(period_dirs[i]["splat"])
                if splat_cloud is not None:
                    pts, cols = splat_cloud
                    save_ply(pts, cols, period_dirs[i]["base"] / "point_cloud_splat.ply")
                    period_final_clouds.append((pts, cols))
                    used_splat = True
                    print(f"  Using Gaussian Splat point cloud for period_{i:02d}")

        if not used_splat:
            print(f"  [fallback] Using depth-based point cloud for period_{i:02d}")
            period_final_clouds.append(period_depth_clouds[i])
    print()

    # ── [5/6] ICP alignment ───────────────────────────────────────────────────
    print("[5/6] Aligning periods with ICP ...")
    try:
        aligned_clouds, transforms = align_point_clouds(period_final_clouds, VOXEL_SIZE)
        all_pts = np.concatenate([p for p, _ in aligned_clouds])
        all_cols = np.concatenate([c for _, c in aligned_clouds])
        save_ply(all_pts, all_cols, OUTPUT_DIR / "aligned_clouds.ply")
    except Exception as e:
        print(f"  [warn] ICP alignment failed ({e}) — using unaligned clouds")
        aligned_clouds = period_final_clouds
    print()

    # ── [6/6] Export 4D HTML viewer ───────────────────────────────────────────
    print("[6/6] Exporting 4D HTML viewer ...")
    html_path = OUTPUT_DIR / "4d_world_model.html"
    export_4d_html(aligned_clouds, labels, html_path)

    print("\n" + "─" * 60)
    print("Done!")
    print(f"  Period clouds:  {PERIODS_DIR}/")
    print(f"  Aligned cloud:  {OUTPUT_DIR}/aligned_clouds.ply")
    print(f"  4D viewer:      {html_path}  ← open in browser")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
