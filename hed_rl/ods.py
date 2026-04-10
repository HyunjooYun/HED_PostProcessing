from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .data import DatasetIndex
from .evaluation import EvaluationRow, summarize_rows
from .metrics import binarize, boundary_f1_score, iou_score, precision_recall_f1


@dataclass(slots=True)
class ODSSweepEntry:
    threshold: float
    precision_mean: float
    recall_mean: float
    f1_mean: float
    boundary_f1_mean: float
    iou_mean: float


@dataclass(slots=True)
class ODSSweepResult:
    best_threshold: float
    best_summary: dict[str, float]
    sweep_entries: list[ODSSweepEntry]


def build_threshold_grid(start: float = 0.05, stop: float = 1.0, step: float = 0.05) -> list[float]:
    if step <= 0:
        raise ValueError("Threshold step must be positive")
    threshold_values: list[float] = []
    current = start
    while current <= stop + 1e-8:
        threshold_values.append(round(current, 6))
        current += step
    if not threshold_values:
        raise ValueError("Threshold grid is empty")
    return threshold_values


def evaluate_hed_threshold(dataset: DatasetIndex, threshold: float) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        binary_map = binarize(sample.hed, threshold).astype(np.float32)
        precision, recall, f1 = precision_recall_f1(binary_map, sample.gt)
        _, _, boundary_f1 = boundary_f1_score(binary_map, sample.gt)
        iou = iou_score(binary_map, sample.gt)
        rows.append(EvaluationRow(sample.sample_id, precision, recall, f1, boundary_f1, iou))
    return rows


def select_ods_threshold(dataset: DatasetIndex, threshold_values: list[float]) -> ODSSweepResult:
    best_threshold = threshold_values[0]
    best_summary: dict[str, float] | None = None
    best_score = float("-inf")
    sweep_entries: list[ODSSweepEntry] = []

    for threshold in threshold_values:
        rows = evaluate_hed_threshold(dataset, threshold)
        summary = summarize_rows(rows)
        score = float(summary.get("boundary_f1_mean", 0.0))
        sweep_entries.append(
            ODSSweepEntry(
                threshold=threshold,
                precision_mean=float(summary.get("precision_mean", 0.0)),
                recall_mean=float(summary.get("recall_mean", 0.0)),
                f1_mean=float(summary.get("f1_mean", 0.0)),
                boundary_f1_mean=score,
                iou_mean=float(summary.get("iou_mean", 0.0)),
            )
        )
        if score > best_score:
            best_threshold = threshold
            best_summary = summary
            best_score = score

    if best_summary is None:
        raise RuntimeError("Failed to select an ODS threshold")

    return ODSSweepResult(best_threshold=best_threshold, best_summary=best_summary, sweep_entries=sweep_entries)


def save_binary_map(binary_map: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((binary_map.astype(np.uint8) * 255), mode="L")
    image.save(output_path)


def export_ods_binary_maps(dataset: DatasetIndex, threshold: float, output_dir: Path) -> tuple[list[EvaluationRow], list[tuple[str, str]]]:
    rows: list[EvaluationRow] = []
    manifest_rows: list[tuple[str, str]] = []
    maps_dir = output_dir / "maps"

    for idx in range(len(dataset)):
        sample = dataset[idx]
        binary_map = binarize(sample.hed, threshold).astype(np.float32)
        output_path = maps_dir / f"{sample.sample_id}_ods_binary.png"
        save_binary_map(binary_map, output_path)

        precision, recall, f1 = precision_recall_f1(binary_map, sample.gt)
        _, _, boundary_f1 = boundary_f1_score(binary_map, sample.gt)
        iou = iou_score(binary_map, sample.gt)
        rows.append(EvaluationRow(sample.sample_id, precision, recall, f1, boundary_f1, iou))
        manifest_rows.append((sample.sample_id, str(output_path)))

    return rows, manifest_rows


def write_ods_sweep_csv(entries: list[ODSSweepEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "precision_mean", "recall_mean", "f1_mean", "boundary_f1_mean", "iou_mean"])
        for entry in entries:
            writer.writerow(
                [
                    entry.threshold,
                    entry.precision_mean,
                    entry.recall_mean,
                    entry.f1_mean,
                    entry.boundary_f1_mean,
                    entry.iou_mean,
                ]
            )


def write_export_manifest(rows: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "output_path"])
        for sample_id, output_path in rows:
            writer.writerow([sample_id, output_path])