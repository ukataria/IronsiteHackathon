# Temporal Construction World Model

This project builds a pipeline that converts a construction walkthrough
video into:

-   Segmented time chunks
-   High-quality selected frames (400--800 total)
-   3D point clouds per segment (via COLMAP)
-   Aligned point clouds in a shared coordinate frame
-   Optional temporal change overlays
-   App-ready assets for a 3D slider visualization

------------------------------------------------------------------------

# What You Will Be Able to See

After running the full pipeline, you will be able to:

-   View the reconstructed 3D point cloud
-   Move a time slider
-   Watch the construction site evolve segment by segment
-   Optionally toggle:
    -   Added geometry (green)
    -   Removed geometry (red)
    -   Baseline/reference map

The slider switches between aligned segment reconstructions.

This is not a continuous animation --- it is a discrete time-indexed 3D
sequence.

------------------------------------------------------------------------

# End-to-End Pipeline Overview

Video\
↓\
Temporal Segmentation\
↓\
Frame Extraction (2--3 FPS)\
↓\
Blur Filtering + Deduplication\
↓\
Select 400--800 High-Quality Frames\
↓\
COLMAP Reconstruction (per segment)\
↓\
Point Cloud Export (.ply)\
↓\
Temporal Alignment (ICP → reference frame)\
↓\
Optional Change Detection\
↓\
Export timeline.json + assets\
↓\
3D Viewer with Slider

------------------------------------------------------------------------

# Output Structure (After Full Run)

data/derived/{video_id}/ ├─ pointclouds/ │ ├─ segment_0\_aligned.ply │
├─ segment_1\_aligned.ply │ ├─ segment_2\_aligned.ply │ └─ reference.ply
├─ changes/ │ ├─ added_1.ply │ ├─ removed_1.ply │ └─ clusters_1.json ├─
transforms/ │ ├─ segment_1\_to_ref.json │ └─ timeline.json

These files are sufficient to power a web-based slider viewer.

------------------------------------------------------------------------

# 3D Temporal Viewer (Slider)

The viewer:

1.  Loads timeline.json
2.  Loads each aligned point cloud
3.  Creates a slider with one step per segment
4.  On slider change:
    -   Removes the current point cloud
    -   Adds the selected segment's point cloud
5.  Optionally overlays change clouds

------------------------------------------------------------------------

# Viewer Folder Structure

app/ ├─ index.html ├─ main.js ├─ timeline.json └─ assets/ ├─
segment_0\_aligned.ply ├─ segment_1\_aligned.ply └─ ...

The app can be static (no backend required).

------------------------------------------------------------------------

# Change Visualization (Optional)

When enabled:

-   Added geometry → colored green
-   Removed geometry → colored red

Computed via nearest-neighbor differencing between aligned point clouds.

------------------------------------------------------------------------

# Future Extensions

-   Semantic labeling (wall, ladder, worker, etc.)
-   2D-to-3D label projection
-   Persistent object tracking
-   Multi-session comparison
-   NeRF / 3D Gaussian Splat rendering

------------------------------------------------------------------------

# Expected MVP Outcome

After running the pipeline on a construction video, you will have:

-   A reconstructed 3D environment
-   Multiple aligned temporal slices
-   A working 3D slider visualization
-   Optional change overlays

This is a functional temporal 3D world model.
