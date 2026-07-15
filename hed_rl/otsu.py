from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import DatasetIndex
from .evaluation import EvaluationRow, summarize_rows
from .metrics import binarize, boundary_f1_score, iou_score, precision_recall_f1
from .ods import save_binary_map, write_export_manifest


@dataclass(slots=True)
class OtsuSampleResult:
    sample_id: str
    threshold: float
    precision: float
    recall: float
    f1: float
    boundary_f1: float
    iou: float
    output_path: str = ""


@dataclass(slots=True)
class OtsuExportResult:
    rows: list[EvaluationRow]
    sample_results: list[OtsuSampleResult]
    manifest_rows: list[tuple[str, str]]
    summary: dict[str, float]


def compute_otsu_threshold(image: np.ndarray, num_bins: int = 256) -> float:
    values = np.clip(image.astype(np.float32).ravel(), 0.0, 1.0)
    if values.size == 0:
        raise ValueError("Cannot compute Otsu threshold for an empty image")

    hist, bin_edges = np.histogram(values, bins=num_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0.0:
        return 0.5

    probabilities = hist / total
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    cumulative_prob = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * bin_centers)
    global_mean = cumulative_mean[-1]

    denominator = cumulative_prob * (1.0 - cumulative_prob)
    numerator = (global_mean * cumulative_prob - cumulative_mean) ** 2
    between_class_variance = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    best_index = int(np.argmax(between_class_variance))
    return float(bin_centers[best_index])


def evaluate_samplewise_otsu_thresholds(dataset: DatasetIndex) -> list[OtsuSampleResult]:
    sample_results: list[OtsuSampleResult] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        threshold = compute_otsu_threshold(sample.hed)
        binary_map = binarize(sample.hed, threshold).astype(np.float32)
        precision, recall, f1 = precision_recall_f1(binary_map, sample.gt)
        _, _, boundary_f1 = boundary_f1_score(binary_map, sample.gt)
        iou = iou_score(binary_map, sample.gt)
        sample_results.append(
            OtsuSampleResult(
                sample_id=sample.sample_id,
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=f1,
                boundary_f1=boundary_f1,
                iou=iou,
            )
        )
    return sample_results


def export_otsu_binary_maps(dataset: DatasetIndex, output_dir: Path) -> OtsuExportResult:
    sample_results = evaluate_samplewise_otsu_thresholds(dataset)
    rows: list[EvaluationRow] = []
    manifest_rows: list[tuple[str, str]] = []
    maps_dir = output_dir / "maps"

    sample_by_id = {dataset.samples[index].sample_id: dataset[index] for index in range(len(dataset))}
    for sample_result in sample_results:
        sample = sample_by_id[sample_result.sample_id]
        binary_map = binarize(sample.hed, sample_result.threshold).astype(np.float32)
        output_path = maps_dir / f"{sample.sample_id}_otsu_binary.png"
        save_binary_map(binary_map, output_path)

        sample_result.output_path = str(output_path)
        rows.append(
            EvaluationRow(
                sample_id=sample_result.sample_id,
                precision=sample_result.precision,
                recall=sample_result.recall,
                f1=sample_result.f1,
                boundary_f1=sample_result.boundary_f1,
                iou=sample_result.iou,
            )
        )
        manifest_rows.append((sample.sample_id, str(output_path)))

    return OtsuExportResult(
        rows=rows,
        sample_results=sample_results,
        manifest_rows=manifest_rows,
        summary=summarize_rows(rows),
    )


def write_otsu_thresholds_csv(sample_results: list[OtsuSampleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "threshold", "precision", "recall", "f1", "boundary_f1", "iou", "output_path"])
        for sample_result in sample_results:
            writer.writerow(
                [
                    sample_result.sample_id,
                    sample_result.threshold,
                    sample_result.precision,
                    sample_result.recall,
                    sample_result.f1,
                    sample_result.boundary_f1,
                    sample_result.iou,
                    sample_result.output_path,
                ]
            )


def summarize_otsu_thresholds(sample_results: list[OtsuSampleResult]) -> dict[str, float]:
    thresholds = np.array([sample_result.threshold for sample_result in sample_results], dtype=np.float32)
    boundary_f1_values = np.array([sample_result.boundary_f1 for sample_result in sample_results], dtype=np.float32)
    iou_values = np.array([sample_result.iou for sample_result in sample_results], dtype=np.float32)
    return {
        "count": float(len(sample_results)),
        "threshold_mean": float(thresholds.mean()),
        "threshold_std": float(thresholds.std()),
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
        "otsu_boundary_f1_mean": float(boundary_f1_values.mean()),
        "otsu_boundary_f1_std": float(boundary_f1_values.std()),
        "otsu_iou_mean": float(iou_values.mean()),
        "otsu_iou_std": float(iou_values.std()),
    }


__all__ = [
    "OtsuExportResult",
    "OtsuSampleResult",
    "compute_otsu_threshold",
    "evaluate_samplewise_otsu_thresholds",
    "export_otsu_binary_maps",
    "summarize_otsu_thresholds",
    "write_otsu_thresholds_csv",
]