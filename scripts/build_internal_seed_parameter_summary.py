from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_policy_row(seed: str, algorithm: str, summary: dict[str, object]) -> dict[str, object]:
    return {
        "seed": seed,
        "algorithm": algorithm,
        "count": summary.get("count", "N/A"),
        "boundary_f1_mean": summary.get("boundary_f1_mean", "N/A"),
        "iou_mean": summary.get("iou_mean", "N/A"),
        "threshold_mean": summary.get("threshold_mean_mean", "N/A"),
        "threshold_std": summary.get("threshold_mean_std", "N/A"),
        "blur_sigma_mean": summary.get("blur_sigma_mean_mean", "N/A"),
        "blur_sigma_std": summary.get("blur_sigma_mean_std", "N/A"),
        "morphology_radius_mean": summary.get("morphology_radius_mean_mean", "N/A"),
        "morphology_radius_std": summary.get("morphology_radius_mean_std", "N/A"),
        "nms_strength_mean": summary.get("nms_strength_mean_mean", "N/A"),
        "nms_strength_std": summary.get("nms_strength_mean_std", "N/A"),
    }


def extract_hed_row(seed: str, summary: dict[str, object]) -> dict[str, object]:
    return {
        "seed": seed,
        "algorithm": "HED",
        "count": summary.get("count", "N/A"),
        "boundary_f1_mean": summary.get("boundary_f1_mean", "N/A"),
        "iou_mean": summary.get("iou_mean", "N/A"),
        "threshold_mean": summary.get("threshold", summary.get("threshold_mean", "N/A")),
        "threshold_std": 0.0,
        "blur_sigma_mean": "N/A",
        "blur_sigma_std": "N/A",
        "morphology_radius_mean": "N/A",
        "morphology_radius_std": "N/A",
        "nms_strength_mean": "N/A",
        "nms_strength_std": "N/A",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a 200-image internal-test per-seed parameter comparison CSV"
    )
    parser.add_argument("--hed-summary", default="runs/hed_test_vs_gt/summary.json")
    parser.add_argument(
        "--seed-runs",
        nargs="+",
        required=True,
        help="Pairs of seed=dqn_summary_json=ppo_summary_json",
    )
    parser.add_argument("--output", default="runs/paper_compare/internal_test_seed_parameter_comparison.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hed_summary = read_json(Path(args.hed_summary))
    rows: list[dict[str, object]] = []

    for seed_run in args.seed_runs:
        seed, dqn_summary_path, ppo_summary_path = seed_run.split("=", 2)
        rows.append(extract_hed_row(seed, hed_summary))
        rows.append(extract_policy_row(seed, "DQN", read_json(Path(dqn_summary_path))))
        rows.append(extract_policy_row(seed, "PPO", read_json(Path(ppo_summary_path))))

    fieldnames = [
        "seed",
        "algorithm",
        "count",
        "boundary_f1_mean",
        "iou_mean",
        "threshold_mean",
        "threshold_std",
        "blur_sigma_mean",
        "blur_sigma_std",
        "morphology_radius_mean",
        "morphology_radius_std",
        "nms_strength_mean",
        "nms_strength_std",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(output_path)


if __name__ == "__main__":
    main()