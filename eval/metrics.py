"""Depth evaluation metrics."""

import numpy as np
from typing import Dict, Optional


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard depth evaluation metrics.

    Args:
        pred: Predicted depth map (H, W) in meters
        gt: Ground truth depth map (H, W) in meters
        valid_mask: Optional boolean mask of valid pixels (H, W)

    Returns:
        Dictionary of metrics:
            - abs_rel: Mean absolute relative error
            - rmse: Root mean squared error
            - mae: Mean absolute error (meters)
            - delta_1: % pixels with max(pred/gt, gt/pred) < 1.25
            - delta_2: % pixels with max(pred/gt, gt/pred) < 1.25^2
            - delta_3: % pixels with max(pred/gt, gt/pred) < 1.25^3
    """
    if valid_mask is None:
        # Create default valid mask (non-zero, finite values)
        valid_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    else:
        valid_mask = valid_mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)

    if valid_mask.sum() == 0:
        return {
            "abs_rel": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "delta_1": np.nan,
            "delta_2": np.nan,
            "delta_3": np.nan,
            "num_valid_pixels": 0
        }

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    # Absolute relative error
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)

    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

    # MAE
    mae = np.mean(np.abs(pred_valid - gt_valid))

    # Delta thresholds
    ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    delta_1 = (ratio < 1.25).mean()
    delta_2 = (ratio < 1.25 ** 2).mean()
    delta_3 = (ratio < 1.25 ** 3).mean()

    return {
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "mae": float(mae),
        "delta_1": float(delta_1),
        "delta_2": float(delta_2),
        "delta_3": float(delta_3),
        "num_valid_pixels": int(valid_mask.sum())
    }


def compute_point_depth_metrics(
    predictions: np.ndarray,
    ground_truths: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics for point-wise depth predictions (e.g., from VLMs).

    Args:
        predictions: Array of predicted depths (N,) in meters
        ground_truths: Array of ground truth depths (N,) in meters

    Returns:
        Dictionary of metrics:
            - abs_rel: Mean absolute relative error
            - rmse: Root mean squared error
            - mae: Mean absolute error (meters)
    """
    # Filter out invalid predictions and ground truths
    valid_mask = (
        np.isfinite(predictions) &
        np.isfinite(ground_truths) &
        (predictions > 0) &
        (ground_truths > 0)
    )

    if valid_mask.sum() == 0:
        return {
            "abs_rel": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "num_valid_points": 0,
            "num_total_points": len(predictions)
        }

    pred_valid = predictions[valid_mask]
    gt_valid = ground_truths[valid_mask]

    # Absolute relative error
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)

    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))

    # MAE
    mae = np.mean(np.abs(pred_valid - gt_valid))

    return {
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "mae": float(mae),
        "num_valid_points": int(valid_mask.sum()),
        "num_total_points": len(predictions)
    }


def compute_scale_aligned_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    scale_method: str = "median"
) -> Dict[str, float]:
    """
    Compute metrics with scale alignment (for models that don't predict metric depth).

    Args:
        pred: Predicted depth map (H, W)
        gt: Ground truth depth map (H, W) in meters
        valid_mask: Optional boolean mask of valid pixels (H, W)
        scale_method: "median" or "least_squares"

    Returns:
        Dictionary with metrics and the scale factor used
    """
    if valid_mask is None:
        valid_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    else:
        valid_mask = valid_mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)

    if valid_mask.sum() == 0:
        return {
            "scale_factor": np.nan,
            **{k: np.nan for k in ["abs_rel", "rmse", "mae", "delta_1", "delta_2", "delta_3"]},
            "num_valid_pixels": 0
        }

    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]

    # Compute scale
    if scale_method == "median":
        scale = np.median(gt_valid) / np.median(pred_valid)
    elif scale_method == "least_squares":
        scale = np.sum(gt_valid * pred_valid) / np.sum(pred_valid ** 2)
    else:
        raise ValueError(f"Unknown scale method: {scale_method}")

    # Scale predictions
    pred_scaled = pred * scale

    # Compute metrics on scaled predictions
    metrics = compute_depth_metrics(pred_scaled, gt, valid_mask)
    metrics["scale_factor"] = float(scale)

    return metrics
