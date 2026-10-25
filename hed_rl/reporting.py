from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import EvaluationRow, OfflineEvaluationRow, PolicyEvaluationRow


ALGORITHM_COLORS: dict[str, str] = {
    "dqn": "#239C23",
    "ppo": "#F57C00",
}

SMOOTHING_FACTOR = 0.7
TRAINING_PLOT_METRICS = [
    "episode_reward_mean",
    "episode_reward_std",
    "episode_length_mean",
    "episode_length_std",
    "train_loss",
]
EVAL_PLOT_METRICS = ["reward_mean", "reward_std", "reward_max"]


def _get_algorithm_color(algorithm: str | None) -> str:
    if algorithm is None:
        return "#33658A"
    return ALGORITHM_COLORS.get(algorithm.lower(), "#33658A")


def _smooth_series(values: np.ndarray, factor: float = SMOOTHING_FACTOR) -> np.ndarray:
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = factor * smoothed[index - 1] + (1.0 - factor) * values[index]
    return smoothed


def write_rows_csv(rows: list[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "precision", "recall", "f1", "boundary_f1", "iou"])
        for row in rows:
            writer.writerow([row.sample_id, row.precision, row.recall, row.f1, row.boundary_f1, row.iou])


def write_summary_plot(summary: dict[str, float], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key in ["precision_mean", "recall_mean", "f1_mean", "boundary_f1_mean", "iou_mean"] if key in summary]
    values = [summary[key] for key in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(keys, values, color=["#33658A", "#55D6BE", "#2F4858", "#F6AE2D", "#F26419"])
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_policy_rows_csv(rows: list[PolicyEvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "reward",
                "steps",
                "threshold_initial",
                "threshold_final",
                "threshold_mean",
                "threshold_std",
                "blur_sigma_initial",
                "blur_sigma_final",
                "blur_sigma_mean",
                "blur_sigma_std",
                "morphology_radius_initial",
                "morphology_radius_final",
                "morphology_radius_mean",
                "morphology_radius_std",
                "nms_strength_initial",
                "nms_strength_final",
                "nms_strength_mean",
                "nms_strength_std",
                "precision",
                "recall",
                "f1",
                "boundary_f1",
                "iou",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.sample_id,
                    row.reward,
                    row.steps,
                    row.threshold_initial,
                    row.threshold_final,
                    row.threshold_mean,
                    row.threshold_std,
                    row.blur_sigma_initial,
                    row.blur_sigma_final,
                    row.blur_sigma_mean,
                    row.blur_sigma_std,
                    row.morphology_radius_initial,
                    row.morphology_radius_final,
                    row.morphology_radius_mean,
                    row.morphology_radius_std,
                    row.nms_strength_initial,
                    row.nms_strength_final,
                    row.nms_strength_mean,
                    row.nms_strength_std,
                    row.precision,
                    row.recall,
                    row.f1,
                    row.boundary_f1,
                    row.iou,
                ]
            )


def write_offline_rows_csv(rows: list[OfflineEvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_names = ("threshold", "blur_sigma", "morphology_radius", "nms_strength")
    header = ["sample_id", "steps"]
    for name in param_names:
        header += [f"{name}_initial", f"{name}_final", f"{name}_mean", f"{name}_std"]
    header += ["precision", "recall", "f1", "boundary_f1", "iou"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            values: list[float | int | str] = [row.sample_id, row.steps]
            for name in param_names:
                values += [row.params_initial[name], row.params_final[name], row.params_mean[name], row.params_std[name]]
            values += [row.precision, row.recall, row.f1, row.boundary_f1, row.iou]
            writer.writerow(values)


def write_metric_records_csv(records: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def read_metric_records_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def merge_metric_records(*record_groups: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for records in record_groups:
        merged.extend(records)
    merged.sort(key=lambda row: (row.get("algorithm", ""), float(row.get("timestep", "0"))))
    return merged


def _sanitize_series(records: list[dict[str, float | str]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    for record in records:
        x_raw = record.get(x_key)
        y_raw = record.get(y_key)
        try:
            x_value = float(x_raw)  # type: ignore[arg-type]
            y_value = float(y_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if np.isnan(x_value) or np.isnan(y_value):
            continue
        x_values.append(x_value)
        y_values.append(y_value)
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def write_single_metric_plot(
    records: list[dict[str, float | str]],
    metric_key: str,
    path: Path,
    title: str,
    algorithm: str | None = None,
) -> None:
    x_values, y_values = _sanitize_series(records, "timestep", metric_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if len(x_values) > 0:
        plt.plot(x_values, _smooth_series(y_values), color=_get_algorithm_color(algorithm), linewidth=2)
    plt.xlabel("timestep")
    plt.ylabel(metric_key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_compare_metric_plot(records: list[dict[str, float | str]], metric_key: str, path: Path, title: str) -> None:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("algorithm", "unknown"))].append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    fallback_palette = ["#33658A", "#F26419", "#55D6BE", "#2F4858"]
    plotted_labels = 0
    for index, (algorithm, items) in enumerate(sorted(grouped.items())):
        x_values, y_values = _sanitize_series(items, "timestep", metric_key)
        if len(x_values) == 0:
            continue
        color = ALGORITHM_COLORS.get(algorithm.lower(), fallback_palette[index % len(fallback_palette)])
        plt.plot(x_values, _smooth_series(y_values), linewidth=2, color=color, label=algorithm.upper())
        plotted_labels += 1
    plt.xlabel("timestep")
    plt.ylabel(metric_key)
    plt.title(title)
    if plotted_labels > 0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_stability_plots(algorithm: str, records: list[dict[str, float | str]], output_dir: Path) -> None:
    metric_names = {
        "dqn": [
            "q_mean",
            "q_max",
            "q_std",
            "td_error_mean",
            "td_error_std",
            "td_error_abs_mean",
            "exploration_rate",
        ],
        "ppo": [
            "explained_variance",
            "approx_kl",
            "clip_fraction",
            "value_loss",
            "entropy_loss",
        ],
    }
    for metric_name in metric_names[algorithm]:
        file_name = f"{algorithm}_stability_{metric_name}.png"
        write_single_metric_plot(
            records,
            metric_name,
            output_dir / file_name,
            title=f"{algorithm.upper()} {metric_name}",
            algorithm=algorithm,
        )


def write_training_plots(records: list[dict[str, float | str]], output_dir: Path) -> list[Path]:
    generated_paths: list[Path] = []
    for metric_name in TRAINING_PLOT_METRICS:
        output_path = output_dir / f"training_{metric_name}.png"
        write_compare_metric_plot(records, metric_name, output_path, title=f"Training {metric_name}")
        generated_paths.append(output_path)
    return generated_paths


def write_eval_plots(records: list[dict[str, float | str]], output_dir: Path) -> list[Path]:
    generated_paths: list[Path] = []
    for metric_name in EVAL_PLOT_METRICS:
        output_path = output_dir / f"eval_compare_{metric_name}.png"
        write_compare_metric_plot(records, metric_name, output_path, title=f"Eval {metric_name}")
        generated_paths.append(output_path)
    return generated_paths