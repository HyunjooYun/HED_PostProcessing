from __future__ import annotations

import numpy as np


def binarize(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (image >= threshold).astype(np.uint8)


def precision_recall_f1(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    true_positive = np.logical_and(pred, tgt).sum(dtype=np.float64)
    false_positive = np.logical_and(pred, np.logical_not(tgt)).sum(dtype=np.float64)
    false_negative = np.logical_and(np.logical_not(pred), tgt).sum(dtype=np.float64)

    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return float(precision), float(recall), float(f1)


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    intersection = np.logical_and(pred, tgt).sum(dtype=np.float64)
    union = np.logical_or(pred, tgt).sum(dtype=np.float64)
    return float(intersection / max(union, 1.0))


def dilate(binary_map: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return binary_map.astype(bool)
    pixel_radius = int(np.ceil(radius))
    squared_radius = radius * radius
    offsets = [
        (row_offset, col_offset)
        for row_offset in range(-pixel_radius, pixel_radius + 1)
        for col_offset in range(-pixel_radius, pixel_radius + 1)
        if row_offset * row_offset + col_offset * col_offset <= squared_radius
    ]
    padded = np.pad(binary_map.astype(bool), pixel_radius, mode="constant")
    height, width = binary_map.shape
    result = np.zeros_like(binary_map, dtype=bool)
    for row_offset, col_offset in offsets:
        window = padded[
            row_offset + pixel_radius : row_offset + pixel_radius + height,
            col_offset + pixel_radius : col_offset + pixel_radius + width,
        ]
        result |= window
    return result


def boundary_f1_score(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: float = np.sqrt(8.0),
) -> tuple[float, float, float]:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    pred_match = dilate(tgt, tolerance)
    tgt_match = dilate(pred, tolerance)
    true_positive_boundary = np.logical_and(pred, pred_match).sum(dtype=np.float64)
    matched_target = np.logical_and(tgt, tgt_match).sum(dtype=np.float64)
    pred_total = pred.sum(dtype=np.float64)
    tgt_total = tgt.sum(dtype=np.float64)

    precision = true_positive_boundary / max(pred_total, 1.0)
    recall = matched_target / max(tgt_total, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return float(precision), float(recall), float(f1)