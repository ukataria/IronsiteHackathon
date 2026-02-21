"""
Generate a 3D world model from construction site video using monocular depth estimation.

Pipeline:
  video → keyframes → Depth Anything V2 → colored point clouds → PLY + HTML viewer

Setup (run once):
    uv init          # if no pyproject.toml yet
    uv add opencv-python torch torchvision transformers pillow numpy open3d plotly

Usage:
    uv run python generate_3d_world.py
    uv run python generate_3d_world.py --video 01_production_masonry.mp4 --frames 6
    uv run python generate_3d_world.py --no-viewer   # headless / Vast.ai
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


# ── Constants ──────────────────────────────────────────────────────────────────

VIDEO_PATH = Path("videoplayback.mp4")
DATA_DIR = Path("data")
FRAMES_DIR = DATA_DIR / "frames"
DEPTH_DIR = DATA_DIR / "depth"
PC_DIR = DATA_DIR / "point_clouds"

# Depth Anything V2 — swap for "Large" on Vast.ai for better quality
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

DEFAULT_NUM_KEYFRAMES = 6
CAMERA_FOV_DEG = 69.0   # wide-angle construction camera estimate
MAX_DEPTH_SCALE = 8.0   # scale relative depth to this many virtual "meters"
SUBSAMPLE = 4           # take every Nth pixel (keeps point count manageable)


# ── Utilities ──────────────────────────────────────────────────────────────────

def setup_dirs() -> None:
    """Create required output directories."""
    for d in [FRAMES_DIR, DEPTH_DIR, PC_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Return the best available compute device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    print("  [warn] No GPU found — depth estimation will be slow on CPU")
    return "cpu"


# ── Frame Extraction ───────────────────────────────────────────────────────────

def extract_keyframes(video_path: Path, num_frames: int) -> list[Path]:
    """
    Extract evenly-spaced keyframes from a video, skipping the first/last 10%.

    Returns a list of saved PNG paths.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_s = total / fps if fps > 0 else 0
    print(f"  Video: {total} frames @ {fps:.1f} fps ({duration_s:.1f}s)")

    start = int(total * 0.10)
    end = int(total * 0.90)
    indices = np.linspace(start, end, num_frames, dtype=int).tolist()

    saved: list[Path] = []
    for idx in indices:
        out_path = FRAMES_DIR / f"frame_{idx:05d}.png"
        if out_path.exists():
            print(f"  [cache] {out_path.name}")
            saved.append(out_path)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            print(f"  [warn]  Could not read frame {idx}")
            continue

        cv2.imwrite(str(out_path), frame)
        print(f"  [saved] {out_path.name}")
        saved.append(out_path)

    cap.release()
    return saved


# ── Depth Estimation ───────────────────────────────────────────────────────────

def load_depth_pipeline(device: str):
    """Load Depth Anything V2 via HuggingFace transformers pipeline."""
    from transformers import pipeline as hf_pipeline  # lazy import

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL_ID,
        device=0 if device == "cuda" else -1,
        torch_dtype=torch_dtype,
    )
    print(f"  Model loaded: {DEPTH_MODEL_ID}")
    return pipe


def estimate_depth(pipe, frame_path: Path) -> np.ndarray:
    """
    Run depth estimation on one frame.

    Returns float32 depth array (H, W), values in [0, MAX_DEPTH_SCALE] virtual meters.
    Higher values = farther from camera.
    Caches .npy to disk.
    """
    npy_path = DEPTH_DIR / f"{frame_path.stem}.npy"
    png_path = DEPTH_DIR / f"{frame_path.stem}.png"

    if npy_path.exists():
        print(f"  [cache] {npy_path.name}")
        return np.load(str(npy_path))

    image = Image.open(frame_path).convert("RGB")
    result = pipe(image)

    # predicted_depth: raw float tensor (H, W) — higher = farther for Depth Anything V2
    raw: torch.Tensor = result["predicted_depth"].squeeze()
    depth_f32 = raw.float().numpy()

    # Normalize to [0, MAX_DEPTH_SCALE]
    d_min, d_max = float(depth_f32.min()), float(depth_f32.max())
    depth_scaled = (depth_f32 - d_min) / (d_max - d_min + 1e-8) * MAX_DEPTH_SCALE

    # Save
    np.save(str(npy_path), depth_scaled)
    vis_u8 = (depth_scaled / MAX_DEPTH_SCALE * 255).astype(np.uint8)
    cv2.imwrite(str(png_path), vis_u8)
    print(f"  [saved] {npy_path.name}")

    return depth_scaled


# ── Point Cloud Construction ───────────────────────────────────────────────────

def pinhole_intrinsics(
    width: int, height: int, fov_deg: float
) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) for an estimated pinhole camera."""
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # assume square pixels
    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy


def build_point_cloud(
    frame_path: Path,
    depth: np.ndarray,
    fov_deg: float = CAMERA_FOV_DEG,
    subsample: int = SUBSAMPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project image pixels to 3D space using depth map.

    Returns:
        points: (N, 3) float32 — XYZ in camera space
        colors: (N, 3) float32 — RGB in [0, 1]
    """
    bgr = cv2.imread(str(frame_path))
    h_img, w_img = bgr.shape[:2]

    # Resize depth to match image if needed
    if depth.shape != (h_img, w_img):
        depth = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    fx, fy, cx, cy = pinhole_intrinsics(w_img, h_img, fov_deg)

    # Build subsampled pixel grid
    ys, xs = np.mgrid[0:h_img:subsample, 0:w_img:subsample]
    ys, xs = ys.flatten(), xs.flatten()
    z = depth[ys, xs]

    # Back-projection: X = (u-cx)*z/fx, Y = (v-cy)*z/fy, Z = z
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Colors: BGR → RGB, normalize
    rgb_vals = bgr[ys, xs][:, ::-1].astype(np.float32) / 255.0
    colors = rgb_vals

    # Remove sky / noise at depth extremes
    valid = (z > 0.05) & (z < MAX_DEPTH_SCALE * 0.98)
    return points[valid], colors[valid]


# ── Save & Export ──────────────────────────────────────────────────────────────

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


def export_html(
    all_pcs: list[tuple[np.ndarray, np.ndarray, str]],
    out_path: Path,
    max_pts_per_frame: int = 60_000,
) -> None:
    """
    Export all point clouds as a single interactive Plotly HTML file.
    Frames are spread along the X axis so they appear side-by-side.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [warn] plotly not installed — skipping HTML export")
        return

    traces = []
    x_offset = 0.0

    for pts, cols, label in all_pcs:
        # Sub-sample for web performance
        step = max(1, len(pts) // max_pts_per_frame)
        p = pts[::step]
        c = cols[::step]

        x_offset_val = p[:, 0].min() if len(p) else 0
        p_shifted = p.copy()
        p_shifted[:, 0] += x_offset - x_offset_val

        hex_colors = [
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            for r, g, b in c
        ]

        traces.append(go.Scatter3d(
            x=p_shifted[:, 0],
            y=p_shifted[:, 2],   # depth as Y axis (forward)
            z=-p_shifted[:, 1],  # invert Y for up-is-up
            mode="markers",
            marker=dict(size=1.2, color=hex_colors, opacity=0.85),
            name=label,
        ))

        # Next frame sits 1.5× the scene width to the right
        scene_width = float(p[:, 0].max() - p[:, 0].min()) if len(p) else 5
        x_offset += scene_width * 1.5

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Ghost Blueprint — 3D World Model",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Depth (Z)",
            zaxis_title="Up (Y)",
            aspectmode="data",
            bgcolor="rgb(10,10,20)",
        ),
        paper_bgcolor="rgb(10,10,20)",
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(str(out_path))
    print(f"  [saved] {out_path.name}")


def open_viewer(all_pcs: list[tuple[np.ndarray, np.ndarray, str]]) -> None:
    """Launch interactive Open3D point cloud viewer."""
    try:
        import open3d as o3d
    except ImportError:
        print("  [warn] open3d not installed — skipping interactive viewer")
        return

    geometries = []
    x_cursor = 0.0

    for pts, cols, label in all_pcs:
        pcd = o3d.geometry.PointCloud()
        shifted = pts.copy()
        scene_width = float(pts[:, 0].max() - pts[:, 0].min()) if len(pts) else 5
        shifted[:, 0] += x_cursor - float(pts[:, 0].min())
        pcd.points = o3d.utility.Vector3dVector(shifted)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        geometries.append(pcd)
        x_cursor += scene_width * 1.5

    print("\nOpening Open3D viewer — press Q to quit, H for help...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Ghost Blueprint — 3D World Model",
        width=1400,
        height=800,
    )


# ── Argument Parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 3D world model from construction site video"
    )
    p.add_argument("--video", type=Path, default=VIDEO_PATH,
                   help="Path to input video (default: 01_production_masonry.mp4)")
    p.add_argument("--frames", type=int, default=DEFAULT_NUM_KEYFRAMES,
                   help="Number of keyframes to extract (default: 6)")
    p.add_argument("--fov", type=float, default=CAMERA_FOV_DEG,
                   help="Estimated camera horizontal FOV in degrees (default: 69)")
    p.add_argument("--subsample", type=int, default=SUBSAMPLE,
                   help="Pixel subsampling factor for point cloud (default: 4)")
    p.add_argument("--model", type=str, default=DEPTH_MODEL_ID,
                   help="HuggingFace model ID for depth estimation")
    p.add_argument("--no-viewer", action="store_true",
                   help="Skip Open3D interactive viewer (use on headless servers)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.video.exists():
        print(f"Error: video not found at '{args.video}'")
        sys.exit(1)

    setup_dirs()
    device = get_device()
    print(f"Device: {device}\n")

    # 1. Extract keyframes
    print(f"[1/4] Extracting {args.frames} keyframes from {args.video.name} ...")
    frame_paths = extract_keyframes(args.video, args.frames)
    if not frame_paths:
        print("No frames extracted — exiting.")
        sys.exit(1)
    print(f"  {len(frame_paths)} keyframes ready\n")

    # 2. Load depth model
    print("[2/4] Loading depth estimation model ...")
    global DEPTH_MODEL_ID
    DEPTH_MODEL_ID = args.model
    depth_pipe = load_depth_pipeline(device)
    print()

    # 3. Depth estimation + point cloud per frame
    print("[3/4] Running depth estimation and building point clouds ...")
    all_pcs: list[tuple[np.ndarray, np.ndarray, str]] = []

    for fp in frame_paths:
        print(f"  {fp.name}:")
        depth = estimate_depth(depth_pipe, fp)
        pts, cols = build_point_cloud(fp, depth, fov_deg=args.fov, subsample=args.subsample)
        ply_path = PC_DIR / f"{fp.stem}.ply"
        save_ply(pts, cols, ply_path)
        all_pcs.append((pts, cols, fp.stem))
    print()

    # 4. Export & visualize
    print("[4/4] Exporting visualizations ...")
    html_path = PC_DIR / "world_model.html"
    export_html(all_pcs, html_path)

    print("\n" + "─" * 60)
    print("Done!")
    print(f"  PLY files:    {PC_DIR}/")
    print(f"  HTML viewer:  {html_path}  ← open this in a browser")
    print("─" * 60 + "\n")

    if not args.no_viewer:
        open_viewer(all_pcs)


if __name__ == "__main__":
    main()
