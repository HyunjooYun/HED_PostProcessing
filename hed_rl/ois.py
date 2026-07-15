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
class OISSampleResult:
    sample_id: str
    best_threshold: float
    precision: float
    recall: float
    f1: float
    boundary_f1: float
    iou: float
    output_path: str = ""


@dataclass(slots=True)
class OISExportResult:
    rows: list[EvaluationRow]
    sample_results: list[OISSampleResult]
    manifest_rows: list[tuple[str, str]]
    summary: dict[str, float]


def evaluate_samplewise_optimal_thresholds(dataset: DatasetIndex, threshold_values: list[float]) -> list[OISSampleResult]:
    sample_results: list[OISSampleResult] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        best_result: OISSampleResult | None = None
        best_score = float("-inf")

        for threshold in threshold_values:
            binary_map = binarize(sample.hed, threshold).astype(np.float32)
            precision, recall, f1 = precision_recall_f1(binary_map, sample.gt)
            _, _, boundary_f1 = boundary_f1_score(binary_map, sample.gt)
            iou = iou_score(binary_map, sample.gt)

            if boundary_f1 > best_score:
                best_score = boundary_f1
                best_result = OISSampleResult(
                    sample_id=sample.sample_id,
                    best_threshold=threshold,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    boundary_f1=boundary_f1,
                    iou=iou,
                )

        if best_result is None:
            raise RuntimeError(f"Failed to select an OIS threshold for sample '{sample.sample_id}'")
        sample_results.append(best_result)

    return sample_results


def export_ois_binary_maps(dataset: DatasetIndex, threshold_values: list[float], output_dir: Path) -> OISExportResult:
    sample_results = evaluate_samplewise_optimal_thresholds(dataset, threshold_values)
    rows: list[EvaluationRow] = []
    manifest_rows: list[tuple[str, str]] = []
    maps_dir = output_dir / "maps"

    sample_by_id = {dataset.samples[index].sample_id: dataset[index] for index in range(len(dataset))}
    for sample_result in sample_results:
        sample = sample_by_id[sample_result.sample_id]
        binary_map = binarize(sample.hed, sample_result.best_threshold).astype(np.float32)
        output_path = maps_dir / f"{sample.sample_id}_ois_binary.png"
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

    return OISExportResult(
        rows=rows,
        sample_results=sample_results,
        manifest_rows=manifest_rows,
        summary=summarize_rows(rows),
    )


def write_ois_thresholds_csv(sample_results: list[OISSampleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "best_threshold", "precision", "recall", "f1", "boundary_f1", "iou", "output_path"])
        for sample_result in sample_results:
            writer.writerow(
                [
                    sample_result.sample_id,
                    sample_result.best_threshold,
                    sample_result.precision,
                    sample_result.recall,
                    sample_result.f1,
                    sample_result.boundary_f1,
                    sample_result.iou,
                    sample_result.output_path,
                ]
            )


def summarize_ois_thresholds(sample_results: list[OISSampleResult]) -> dict[str, float]:
    thresholds = np.array([sample_result.best_threshold for sample_result in sample_results], dtype=np.float32)
    boundary_f1_values = np.array([sample_result.boundary_f1 for sample_result in sample_results], dtype=np.float32)
    return {
        "count": float(len(sample_results)),
        "threshold_mean": float(thresholds.mean()),
        "threshold_std": float(thresholds.std()),
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
        "ois_boundary_f1_mean": float(boundary_f1_values.mean()),
        "ois_boundary_f1_std": float(boundary_f1_values.std()),
    }


__all__ = [
    "OISExportResult",
    "OISSampleResult",
    "evaluate_samplewise_optimal_thresholds",
    "export_ois_binary_maps",
    "summarize_ois_thresholds",
    "write_export_manifest",
    "write_ois_thresholds_csv",
]