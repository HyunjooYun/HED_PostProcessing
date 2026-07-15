from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hed_rl.config import PostProcessConfig
from hed_rl.data import Sample, load_image
from hed_rl.env import HEDEdgeEnv
from hed_rl.inference import load_model
from hed_rl.metrics import boundary_f1_score, iou_score, precision_recall_f1


@dataclass(slots=True)
class EvaluationPaths:
    image_dir: Path
    gt_dir: Path
    hed_dir: Path
    hed_binary_dir: Path
    dqn_model_path: Path
    ppo_model_path: Path
    output_dir: Path
    hed_threshold: float = 0.5


class BipedDataset:
    def __init__(self, image_dir: Path, hed_dir: Path, gt_dir: Path):
        sample_ids: list[str] = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            sample_id = image_path.stem
            if (hed_dir / f"{sample_id}_edge.png").exists() and (gt_dir / f"{sample_id}.png").exists():
                sample_ids.append(sample_id)
        if not sample_ids:
            raise RuntimeError("No matched BIPED samples found")
        self.sample_ids = sample_ids
        self.image_dir = image_dir
        self.hed_dir = hed_dir
        self.gt_dir = gt_dir

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Sample:
        sample_id = self.sample_ids[index]
        return Sample(
            sample_id=sample_id,
            image=load_image(self.image_dir / f"{sample_id}.jpg", grayscale=False),
            hed=load_image(self.hed_dir / f"{sample_id}_edge.png", grayscale=True),
            gt=load_image(self.gt_dir / f"{sample_id}.png", grayscale=True),
        )


def save_binary_map(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array.astype(np.uint8) * 255), mode="L").save(path)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_baseline(dataset: BipedDataset, hed_binary_dir: Path, hed_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id in dataset.sample_ids:
        gt = load_image(dataset.gt_dir / f"{sample_id}.png", grayscale=True)
        binary = load_image(hed_binary_dir / f"{sample_id}_ods_binary.png", grayscale=True)
        precision, recall, f1 = precision_recall_f1(binary, gt)
        _, _, boundary_f1 = boundary_f1_score(binary, gt)
        iou = iou_score(binary, gt)
        rows.append(
            {
                "sample_id": sample_id,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "boundary_f1": float(boundary_f1),
                "iou": float(iou),
                "threshold": float(hed_threshold),
            }
        )
    return rows


def evaluate_model(model_path: Path, dataset: BipedDataset, algorithm: str, binary_dir: Path) -> list[dict[str, object]]:
    model = load_model(model_path)
    env = HEDEdgeEnv(dataset, PostProcessConfig(), cycle_images=False)
    rows: list[dict[str, object]] = []
    env.restart_sequence()
    observation, reset_info = env.reset()

    for episode_index in range(len(dataset)):
        reward_history: list[float] = []
        threshold_history = [float(reset_info["params"]["threshold"])]
        blur_sigma_history = [float(reset_info["params"]["blur_sigma"])]
        morphology_radius_history = [float(reset_info["params"]["morphology_radius"])]
        nms_strength_history = [float(reset_info["params"]["nms_strength"])]
        done = False
        truncated = False
        final_info = reset_info
        while not done and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, final_info = env.step(int(action))
            reward_history.append(float(reward))
            threshold_history.append(float(final_info["params"]["threshold"]))
            blur_sigma_history.append(float(final_info["params"]["blur_sigma"]))
            morphology_radius_history.append(float(final_info["params"]["morphology_radius"]))
            nms_strength_history.append(float(final_info["params"]["nms_strength"]))

        sample = env.sample
        processed = env.processed_map
        if sample is None or processed is None:
            raise RuntimeError(f"{algorithm}: environment did not expose final sample output")

        output_path = binary_dir / f"{sample.sample_id}_{algorithm}_binary.png"
        save_binary_map(processed, output_path)

        rewards = np.asarray(reward_history, dtype=np.float32)
        thresholds = np.asarray(threshold_history, dtype=np.float32)
        blur_sigmas = np.asarray(blur_sigma_history, dtype=np.float32)
        morphology_radii = np.asarray(morphology_radius_history, dtype=np.float32)
        nms_strengths = np.asarray(nms_strength_history, dtype=np.float32)
        metrics = final_info["metrics"]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "reward_total": float(rewards.sum()),
                "reward_mean": float(rewards.mean()) if len(rewards) else 0.0,
                "reward_max": float(rewards.max()) if len(rewards) else 0.0,
                "reward_std": float(rewards.std()) if len(rewards) else 0.0,
                "steps": int(len(rewards)),
                "threshold_initial": float(thresholds[0]),
                "threshold_final": float(thresholds[-1]),
                "threshold_mean": float(thresholds.mean()),
                "threshold_std": float(thresholds.std()),
                "blur_sigma_initial": float(blur_sigmas[0]),
                "blur_sigma_final": float(blur_sigmas[-1]),
                "blur_sigma_mean": float(blur_sigmas.mean()),
                "blur_sigma_std": float(blur_sigmas.std()),
                "morphology_radius_initial": float(morphology_radii[0]),
                "morphology_radius_final": float(morphology_radii[-1]),
                "morphology_radius_mean": float(morphology_radii.mean()),
                "morphology_radius_std": float(morphology_radii.std()),
                "nms_strength_initial": float(nms_strengths[0]),
                "nms_strength_final": float(nms_strengths[-1]),
                "nms_strength_mean": float(nms_strengths.mean()),
                "nms_strength_std": float(nms_strengths.std()),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "boundary_f1": float(metrics["boundary_f1"]),
                "iou": float(metrics["iou"]),
                "binary_map": str(output_path),
            }
        )

        if episode_index + 1 < len(dataset):
            observation, reset_info = env.reset()

    return rows


def summarize_policy(rows: list[dict[str, object]], algorithm: str) -> dict[str, object]:
    metric_keys = [
        "reward_total",
        "reward_mean",
        "reward_max",
        "reward_std",
        "steps",
        "threshold_mean",
        "threshold_std",
        "blur_sigma_mean",
        "blur_sigma_std",
        "morphology_radius_mean",
        "morphology_radius_std",
        "nms_strength_mean",
        "nms_strength_std",
        "boundary_f1",
        "iou",
        "f1",
        "precision",
        "recall",
    ]
    summary: dict[str, object] = {"algorithm": algorithm, "count": len(rows)}
    for key in metric_keys:
        values = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def summarize_baseline(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "algorithm": "HED",
        "count": len(rows),
        "boundary_f1_mean": float(np.mean([float(row["boundary_f1"]) for row in rows])),
        "boundary_f1_std": float(np.std([float(row["boundary_f1"]) for row in rows])),
        "iou_mean": float(np.mean([float(row["iou"]) for row in rows])),
        "iou_std": float(np.std([float(row["iou"]) for row in rows])),
        "threshold_mean": float(np.mean([float(row["threshold"]) for row in rows])),
        "threshold_std": float(np.std([float(row["threshold"]) for row in rows])),
    }


def plot_three_series(
    x_values: np.ndarray,
    hed_values: list[float],
    dqn_values: list[float],
    ppo_values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
    y_limit: tuple[float, float] = (0.0, 1.0),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.8))
    plt.plot(x_values, hed_values, color="blue", linewidth=1.6, label="HED")
    plt.plot(x_values, dqn_values, color="green", linewidth=1.2, label="DQN")
    plt.plot(x_values, ppo_values, color="#F57C00", linewidth=1.2, label="PPO")
    plt.xlabel("episode index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(*y_limit)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate seed-specific DQN/PPO models on external BIPED inputs")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--hed-dir", required=True)
    parser.add_argument("--hed-binary-dir", required=True)
    parser.add_argument("--dqn-model-path", required=True)
    parser.add_argument("--ppo-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hed-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = EvaluationPaths(
        image_dir=Path(args.image_dir),
        gt_dir=Path(args.gt_dir),
        hed_dir=Path(args.hed_dir),
        hed_binary_dir=Path(args.hed_binary_dir),
        dqn_model_path=Path(args.dqn_model_path),
        ppo_model_path=Path(args.ppo_model_path),
        output_dir=Path(args.output_dir),
        hed_threshold=args.hed_threshold,
    )

    dataset = BipedDataset(paths.image_dir, paths.hed_dir, paths.gt_dir)
    dqn_binary_dir = paths.output_dir / "dqn_binary_maps"
    ppo_binary_dir = paths.output_dir / "ppo_binary_maps"
    plots_dir = paths.output_dir / "plots"
    for path in [paths.output_dir, dqn_binary_dir, ppo_binary_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    compare_csv_name = f"{paths.output_dir.name}_policy_compare_rows.csv"

    hed_rows = evaluate_baseline(dataset, paths.hed_binary_dir, paths.hed_threshold)
    dqn_rows = evaluate_model(paths.dqn_model_path, dataset, "dqn", dqn_binary_dir)
    ppo_rows = evaluate_model(paths.ppo_model_path, dataset, "ppo", ppo_binary_dir)

    hed_by_id = {str(row["sample_id"]): row for row in hed_rows}
    dqn_by_id = {str(row["sample_id"]): row for row in dqn_rows}
    ppo_by_id = {str(row["sample_id"]): row for row in ppo_rows}

    combined_rows: list[dict[str, object]] = []
    for sample_id in dataset.sample_ids:
        hed = hed_by_id[sample_id]
        dqn = dqn_by_id[sample_id]
        ppo = ppo_by_id[sample_id]
        combined_rows.append(
            {
                "sample_id": sample_id,
                "hed_threshold": hed["threshold"],
                "hed_boundary_f1": hed["boundary_f1"],
                "hed_iou": hed["iou"],
                "dqn_reward_mean": dqn["reward_mean"],
                "dqn_reward_max": dqn["reward_max"],
                "dqn_reward_std": dqn["reward_std"],
                "dqn_threshold_mean": dqn["threshold_mean"],
                "dqn_threshold_std": dqn["threshold_std"],
                "dqn_threshold_final": dqn["threshold_final"],
                "dqn_blur_sigma_mean": dqn["blur_sigma_mean"],
                "dqn_blur_sigma_std": dqn["blur_sigma_std"],
                "dqn_blur_sigma_final": dqn["blur_sigma_final"],
                "dqn_morphology_radius_mean": dqn["morphology_radius_mean"],
                "dqn_morphology_radius_std": dqn["morphology_radius_std"],
                "dqn_morphology_radius_final": dqn["morphology_radius_final"],
                "dqn_nms_strength_mean": dqn["nms_strength_mean"],
                "dqn_nms_strength_std": dqn["nms_strength_std"],
                "dqn_nms_strength_final": dqn["nms_strength_final"],
                "dqn_boundary_f1": dqn["boundary_f1"],
                "dqn_iou": dqn["iou"],
                "ppo_reward_mean": ppo["reward_mean"],
                "ppo_reward_max": ppo["reward_max"],
                "ppo_reward_std": ppo["reward_std"],
                "ppo_threshold_mean": ppo["threshold_mean"],
                "ppo_threshold_std": ppo["threshold_std"],
                "ppo_threshold_final": ppo["threshold_final"],
                "ppo_blur_sigma_mean": ppo["blur_sigma_mean"],
                "ppo_blur_sigma_std": ppo["blur_sigma_std"],
                "ppo_blur_sigma_final": ppo["blur_sigma_final"],
                "ppo_morphology_radius_mean": ppo["morphology_radius_mean"],
                "ppo_morphology_radius_std": ppo["morphology_radius_std"],
                "ppo_morphology_radius_final": ppo["morphology_radius_final"],
                "ppo_nms_strength_mean": ppo["nms_strength_mean"],
                "ppo_nms_strength_std": ppo["nms_strength_std"],
                "ppo_nms_strength_final": ppo["nms_strength_final"],
                "ppo_boundary_f1": ppo["boundary_f1"],
                "ppo_iou": ppo["iou"],
            }
        )

    write_csv(paths.output_dir / "hed_rows.csv", hed_rows, list(hed_rows[0].keys()))
    write_csv(paths.output_dir / "dqn_rows.csv", dqn_rows, list(dqn_rows[0].keys()))
    write_csv(paths.output_dir / "ppo_rows.csv", ppo_rows, list(ppo_rows[0].keys()))
    write_csv(
        paths.output_dir / compare_csv_name,
        combined_rows,
        list(combined_rows[0].keys()),
    )

    summary_rows = [summarize_baseline(hed_rows), summarize_policy(dqn_rows, "DQN"), summarize_policy(ppo_rows, "PPO")]
    summary_fieldnames = sorted({field for row in summary_rows for field in row.keys()})
    write_csv(paths.output_dir / "summary.csv", summary_rows, summary_fieldnames)
    (paths.output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    x_values = np.arange(len(dataset.sample_ids), dtype=np.int32)
    plot_three_series(
        x_values,
        [float(hed_by_id[sample_id]["threshold"]) for sample_id in dataset.sample_ids],
        [float(dqn_by_id[sample_id]["threshold_mean"]) for sample_id in dataset.sample_ids],
        [float(ppo_by_id[sample_id]["threshold_mean"]) for sample_id in dataset.sample_ids],
        ylabel="threshold",
        title="Threshold value comparison: HED vs DQN vs PPO",
        output_path=plots_dir / "threshold_value_comparison.png",
    )
    plot_three_series(
        x_values,
        [float(hed_by_id[sample_id]["boundary_f1"]) for sample_id in dataset.sample_ids],
        [float(dqn_by_id[sample_id]["boundary_f1"]) for sample_id in dataset.sample_ids],
        [float(ppo_by_id[sample_id]["boundary_f1"]) for sample_id in dataset.sample_ids],
        ylabel="boundary_f1",
        title="Per-image boundary F1 scores: HED vs DQN vs PPO",
        output_path=plots_dir / "per_image_boundary_f1_scores.png",
    )
    plot_three_series(
        x_values,
        [float(hed_by_id[sample_id]["iou"]) for sample_id in dataset.sample_ids],
        [float(dqn_by_id[sample_id]["iou"]) for sample_id in dataset.sample_ids],
        [float(ppo_by_id[sample_id]["iou"]) for sample_id in dataset.sample_ids],
        ylabel="iou",
        title="Per-image IoU scores: HED vs DQN vs PPO",
        output_path=plots_dir / "per_image_iou_scores.png",
    )

    print(
        json.dumps(
            {
                "count": len(dataset.sample_ids),
                "output_dir": str(paths.output_dir),
                "combined_csv": str(paths.output_dir / compare_csv_name),
                "summary_csv": str(paths.output_dir / "summary.csv"),
                "dqn_binary_dir": str(dqn_binary_dir),
                "ppo_binary_dir": str(ppo_binary_dir),
                "plots": [
                    str(plots_dir / "threshold_value_comparison.png"),
                    str(plots_dir / "per_image_boundary_f1_scores.png"),
                    str(plots_dir / "per_image_iou_scores.png"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()