from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_row(seed: str, algorithm: str, summary_row: dict[str, object]) -> dict[str, object]:
    if algorithm == "HED":
        return {
            "seed": seed,
            "algorithm": algorithm,
            "boundary_f1_mean": summary_row.get("boundary_f1_mean", "N/A"),
            "iou_mean": summary_row.get("iou_mean", "N/A"),
            "threshold_mean": summary_row.get("threshold_mean", "N/A"),
            "threshold_std": summary_row.get("threshold_std", "N/A"),
            "blur_sigma_mean": "N/A",
            "blur_sigma_std": "N/A",
            "morphology_radius_mean": "N/A",
            "morphology_radius_std": "N/A",
            "nms_strength_mean": "N/A",
            "nms_strength_std": "N/A",
        }
    return {
        "seed": seed,
        "algorithm": algorithm,
        "boundary_f1_mean": summary_row.get("boundary_f1_mean", "N/A"),
        "iou_mean": summary_row.get("iou_mean", "N/A"),
        "threshold_mean": summary_row.get("threshold_mean_mean", "N/A"),
        "threshold_std": summary_row.get("threshold_mean_std", "N/A"),
        "blur_sigma_mean": summary_row.get("blur_sigma_mean_mean", "N/A"),
        "blur_sigma_std": summary_row.get("blur_sigma_mean_std", "N/A"),
        "morphology_radius_mean": summary_row.get("morphology_radius_mean_mean", "N/A"),
        "morphology_radius_std": summary_row.get("morphology_radius_mean_std", "N/A"),
        "nms_strength_mean": summary_row.get("nms_strength_mean_mean", "N/A"),
        "nms_strength_std": summary_row.get("nms_strength_mean_std", "N/A"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-seed parameter comparison CSV from BIPED comparison summaries")
    parser.add_argument("--seed-dirs", nargs="+", required=True, help="Pairs of seed=summary_json_path")
    parser.add_argument("--output", default="runs/paper_compare/seed_parameter_comparison.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed_dir in args.seed_dirs:
        seed, summary_path_str = seed_dir.split("=", 1)
        summary_rows = read_json(Path(summary_path_str))
        for summary_row in summary_rows:
            algorithm = str(summary_row.get("algorithm", ""))
            rows.append(extract_row(seed, algorithm, summary_row))

    fieldnames = [
        "seed",
        "algorithm",
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