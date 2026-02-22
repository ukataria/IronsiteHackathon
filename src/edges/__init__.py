"""Edge detection pipeline stage — Canny edges for VLM augmentation."""

from src.edges.canny import compute_canny_edges

__all__ = ["compute_canny_edges"]
