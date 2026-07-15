from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_policy_compare_rows(rows: list[dict[str, str]], prefix: str) -> dict[str, float]:
    metrics = {
        "boundary_f1_mean": float(np.mean([float(row[f"{prefix}_boundary_f1"]) for row in rows])),
        "boundary_f1_std": float(np.std([float(row[f"{prefix}_boundary_f1"]) for row in rows])),
        "iou_mean": float(np.mean([float(row[f"{prefix}_iou"]) for row in rows])),
        "iou_std": float(np.std([float(row[f"{prefix}_iou"]) for row in rows])),
    }
    f1_key = f"{prefix}_f1"
    if f1_key in rows[0]:
        metrics["f1_mean"] = float(np.mean([float(row[f1_key]) for row in rows]))
        metrics["f1_std"] = float(np.std([float(row[f1_key]) for row in rows]))
    return metrics


def build_method_rows(
    ods_summary: dict[str, object],
    ods_metadata: dict[str, object],
    ois_summary: dict[str, object],
    ois_threshold_summary: dict[str, object],
    policy_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    dqn_summary = summarize_policy_compare_rows(policy_rows, "dqn")
    ppo_summary = summarize_policy_compare_rows(policy_rows, "ppo")

    return [
        {
            "method": "HED ODS",
            "role": "practical baseline",
            "threshold_policy": "single dataset-level threshold",
            "gt_at_test": "No",
            "boundary_f1_mean": float(ods_summary["boundary_f1_mean"]),
            "boundary_f1_std": "N/A",
            "iou_mean": float(ods_summary["iou_mean"]),
            "iou_std": "N/A",
            "threshold_mean": float(ods_metadata["best_threshold"]),
            "threshold_std": 0.0,
            "threshold_min": float(ods_metadata["best_threshold"]),
            "threshold_max": float(ods_metadata["best_threshold"]),
            "notes": "Validation-selected single threshold applied to all test images",
        },
        {
            "method": "HED OIS",
            "role": "oracle upper bound",
            "threshold_policy": "per-image optimal threshold",
            "gt_at_test": "Yes",
            "boundary_f1_mean": float(ois_summary["boundary_f1_mean"]),
            "boundary_f1_std": float(ois_threshold_summary["ois_boundary_f1_std"]),
            "iou_mean": float(ois_summary["iou_mean"]),
            "iou_std": "N/A",
            "threshold_mean": float(ois_threshold_summary["threshold_mean"]),
            "threshold_std": float(ois_threshold_summary["threshold_std"]),
            "threshold_min": float(ois_threshold_summary["threshold_min"]),
            "threshold_max": float(ois_threshold_summary["threshold_max"]),
            "notes": "Per-image GT-aware optimal threshold on test split",
        },
        {
            "method": "DQN",
            "role": "learned adaptive method",
            "threshold_policy": "policy-controlled post-processing",
            "gt_at_test": "No",
            "boundary_f1_mean": dqn_summary["boundary_f1_mean"],
            "boundary_f1_std": dqn_summary["boundary_f1_std"],
            "iou_mean": dqn_summary["iou_mean"],
            "iou_std": dqn_summary["iou_std"],
            "threshold_mean": "N/A",
            "threshold_std": "N/A",
            "threshold_min": "N/A",
            "threshold_max": "N/A",
            "notes": "Threshold traces were not persisted in the existing seed comparison export",
        },
        {
            "method": "PPO",
            "role": "learned adaptive method",
            "threshold_policy": "policy-controlled post-processing",
            "gt_at_test": "No",
            "boundary_f1_mean": ppo_summary["boundary_f1_mean"],
            "boundary_f1_std": ppo_summary["boundary_f1_std"],
            "iou_mean": ppo_summary["iou_mean"],
            "iou_std": ppo_summary["iou_std"],
            "threshold_mean": "N/A",
            "threshold_std": "N/A",
            "threshold_min": "N/A",
            "threshold_max": "N/A",
            "notes": "Threshold traces were not persisted in the existing seed comparison export",
        },
    ]


def build_sample_rows(
    ods_rows: list[dict[str, str]],
    ois_rows: list[dict[str, str]],
    policy_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    ods_by_id = {row["sample_id"]: row for row in ods_rows}
    ois_by_id = {row["sample_id"]: row for row in ois_rows}
    merged_rows: list[dict[str, object]] = []

    for policy_row in policy_rows:
        sample_id = policy_row["sample_id"]
        ods_row = ods_by_id.get(sample_id)
        ois_row = ois_by_id.get(sample_id)
        if ods_row is None or ois_row is None:
            continue
        merged_rows.append(
            {
                "sample_id": sample_id,
                "ods_boundary_f1": float(ods_row["boundary_f1"]),
                "ois_boundary_f1": float(ois_row["boundary_f1"]),
                "dqn_boundary_f1": float(policy_row["dqn_boundary_f1"]),
                "ppo_boundary_f1": float(policy_row["ppo_boundary_f1"]),
                "ods_iou": float(ods_row["iou"]),
                "ois_iou": float(ois_row["iou"]),
                "dqn_iou": float(policy_row["dqn_iou"]),
                "ppo_iou": float(policy_row["ppo_iou"]),
                "ods_threshold": float(ods_row.get("threshold", 0.0)) if "threshold" in ods_row else None,
                "ois_threshold": float(ois_row.get("best_threshold", 0.0)) if "best_threshold" in ois_row else None,
            }
        )
    return merged_rows


def write_markdown_table(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "Method",
        "Role",
        "Threshold Policy",
        "GT at Test",
        "Boundary F1 Mean",
        "Boundary F1 Std",
        "IoU Mean",
        "IoU Std",
        "Threshold Mean",
        "Threshold Std",
        "Threshold Min",
        "Threshold Max",
        "Notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["role"]),
                    str(row["threshold_policy"]),
                    str(row["gt_at_test"]),
                    str(row["boundary_f1_mean"]),
                    str(row["boundary_f1_std"]),
                    str(row["iou_mean"]),
                    str(row["iou_std"]),
                    str(row["threshold_mean"]),
                    str(row["threshold_std"]),
                    str(row["threshold_min"]),
                    str(row["threshold_max"]),
                    str(row["notes"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parameter_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_by_algorithm = {str(row["algorithm"]): row for row in summary_rows}
    hed_row = summary_by_algorithm.get("HED", {})
    dqn_row = summary_by_algorithm.get("DQN", {})
    ppo_row = summary_by_algorithm.get("PPO", {})

    return [
        {
            "method": "HED ODS",
            "threshold": hed_row.get("threshold_mean", "N/A"),
            "blur_sigma": "N/A",
            "morphology_radius": "N/A",
            "nms_strength": "N/A",
            "notes": "Raw HED thresholding baseline without additional RL-controlled post-processing",
        },
        {
            "method": "DQN",
            "threshold": f"{dqn_row.get('threshold_mean_mean', 'N/A')} ± {dqn_row.get('threshold_mean_std', 'N/A')}",
            "blur_sigma": f"{dqn_row.get('blur_sigma_mean_mean', 'N/A')} ± {dqn_row.get('blur_sigma_mean_std', 'N/A')}",
            "morphology_radius": f"{dqn_row.get('morphology_radius_mean_mean', 'N/A')} ± {dqn_row.get('morphology_radius_mean_std', 'N/A')}",
            "nms_strength": f"{dqn_row.get('nms_strength_mean_mean', 'N/A')} ± {dqn_row.get('nms_strength_mean_std', 'N/A')}",
            "notes": "Mean ± std of per-image parameter means across the BIPED test set",
        },
        {
            "method": "PPO",
            "threshold": f"{ppo_row.get('threshold_mean_mean', 'N/A')} ± {ppo_row.get('threshold_mean_std', 'N/A')}",
            "blur_sigma": f"{ppo_row.get('blur_sigma_mean_mean', 'N/A')} ± {ppo_row.get('blur_sigma_mean_std', 'N/A')}",
            "morphology_radius": f"{ppo_row.get('morphology_radius_mean_mean', 'N/A')} ± {ppo_row.get('morphology_radius_mean_std', 'N/A')}",
            "nms_strength": f"{ppo_row.get('nms_strength_mean_mean', 'N/A')} ± {ppo_row.get('nms_strength_mean_std', 'N/A')}",
            "notes": "Mean ± std of per-image parameter means across the BIPED test set",
        },
    ]


def write_parameter_markdown_table(path: Path, rows: list[dict[str, object]]) -> None:
    headers = ["Method", "Threshold", "Blur Sigma", "Morphology Radius", "NMS Strength", "Notes"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["threshold"]),
                    str(row["blur_sigma"]),
                    str(row["morphology_radius"]),
                    str(row["nms_strength"]),
                    str(row["notes"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper-ready ODS/OIS/DQN/PPO comparison table")
    parser.add_argument("--ods-summary", default="runs/ods_binary_maps/test/summary.json")
    parser.add_argument("--ods-metadata", default="runs/ods_binary_maps/ods_threshold.json")
    parser.add_argument("--ods-rows", default="runs/ods_binary_maps/test/rows.csv")
    parser.add_argument("--ois-summary", default="runs/ois_binary_maps/test/summary.json")
    parser.add_argument("--ois-threshold-summary", default="runs/ois_binary_maps/test/threshold_summary.json")
    parser.add_argument("--ois-rows", default="runs/ois_binary_maps/test/ois_thresholds.csv")
    parser.add_argument("--policy-compare-rows", default="runs/seed486_compare/test_sample_metric_compare.csv")
    parser.add_argument("--biped-summary", default="runs/biped_seed486_compare/summary.json")
    parser.add_argument("--output-dir", default="runs/paper_compare")
    args = parser.parse_args()

    ods_summary = read_json(Path(args.ods_summary))
    ods_metadata = read_json(Path(args.ods_metadata))
    ods_rows = read_csv_rows(Path(args.ods_rows))
    ois_summary = read_json(Path(args.ois_summary))
    ois_threshold_summary = read_json(Path(args.ois_threshold_summary))
    ois_rows = read_csv_rows(Path(args.ois_rows))
    policy_rows = read_csv_rows(Path(args.policy_compare_rows))
    biped_summary_rows = read_json(Path(args.biped_summary))

    method_rows = build_method_rows(ods_summary, ods_metadata, ois_summary, ois_threshold_summary, policy_rows)
    sample_rows = build_sample_rows(ods_rows, ois_rows, policy_rows)
    parameter_rows = build_parameter_rows(biped_summary_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "method_comparison_summary.csv", method_rows, list(method_rows[0].keys()))
    write_csv(output_dir / "test_sample_metric_compare_full.csv", sample_rows, list(sample_rows[0].keys()))
    write_csv(output_dir / "parameter_comparison_summary.csv", parameter_rows, list(parameter_rows[0].keys()))
    write_markdown_table(output_dir / "method_comparison_summary.md", method_rows)
    write_parameter_markdown_table(output_dir / "parameter_comparison_summary.md", parameter_rows)

    print(
        json.dumps(
            {
                "method_summary_csv": str(output_dir / "method_comparison_summary.csv"),
                "method_summary_md": str(output_dir / "method_comparison_summary.md"),
                "parameter_summary_csv": str(output_dir / "parameter_comparison_summary.csv"),
                "parameter_summary_md": str(output_dir / "parameter_comparison_summary.md"),
                "sample_compare_csv": str(output_dir / "test_sample_metric_compare_full.csv"),
                "count": len(sample_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()