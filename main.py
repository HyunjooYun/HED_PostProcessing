from __future__ import annotations

import argparse
import json
from pathlib import Path

from hed_rl.config import ExperimentConfig, ParameterState, PostProcessConfig
from hed_rl.data import DatasetIndex
from hed_rl.evaluation import evaluate_fixed_threshold, evaluate_policy_model, summarize_policy_rows, summarize_rows
from hed_rl.inference import export_policy_samples
from hed_rl.ods import build_threshold_grid, export_ods_binary_maps, select_ods_threshold, write_export_manifest, write_ods_sweep_csv
from hed_rl.reporting import read_metric_records_csv, write_compare_metric_plot, write_eval_plots, write_metric_records_csv, write_policy_rows_csv, write_rows_csv, write_summary_plot, write_training_plots, merge_metric_records
from hed_rl.training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HED post-processing RL experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run fixed-parameter baseline evaluation")
    baseline.add_argument("--split", default="test", choices=["train", "val", "test"])
    baseline.add_argument("--threshold", type=float, default=0.5)
    baseline.add_argument("--blur-sigma", type=float, default=1.0)
    baseline.add_argument("--morphology-radius", type=int, default=1)
    baseline.add_argument("--nms-strength", type=float, default=0.5)
    baseline.add_argument("--output-dir", default="runs/baseline")

    train = subparsers.add_parser("train", help="Train an RL agent with SB3")
    train.add_argument("--algorithm", required=True, choices=["dqn", "ppo"])
    train.add_argument("--timesteps", type=int, default=200_000)
    train.add_argument("--run-name", default=None)

    evaluate = subparsers.add_parser("evaluate-model", help="Evaluate a trained DQN or PPO model on a dataset split")
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument("--output-dir", required=True)

    compare = subparsers.add_parser("compare-runs", help="Compare saved DQN and PPO training/evaluation CSV logs")
    compare.add_argument("--dqn-run-dir", required=True)
    compare.add_argument("--ppo-run-dir", required=True)
    compare.add_argument("--output-dir", required=True)

    samples = subparsers.add_parser("export-samples", help="Export qualitative sample images from a trained model")
    samples.add_argument("--model-path", required=True)
    samples.add_argument("--split", default="test", choices=["train", "val", "test"])
    samples.add_argument("--num-samples", type=int, default=10)
    samples.add_argument("--output-dir", required=True)

    ods = subparsers.add_parser("export-ods-binary", help="Select an ODS threshold on validation data and export binary edge maps")
    ods.add_argument("--selection-split", default="val", choices=["train", "val", "test"])
    ods.add_argument("--target-splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ods.add_argument("--threshold-start", type=float, default=0.05)
    ods.add_argument("--threshold-stop", type=float, default=1.0)
    ods.add_argument("--threshold-step", type=float, default=0.05)
    ods.add_argument("--output-dir", default="runs/ods_binary_maps")

    return parser


def run_baseline(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    dataset = DatasetIndex(config.inputs_dir, args.split)
    params = ParameterState(
        threshold=args.threshold,
        blur_sigma=args.blur_sigma,
        morphology_radius=args.morphology_radius,
        nms_strength=args.nms_strength,
    ).clip()
    rows = evaluate_fixed_threshold(dataset, params)
    summary = summarize_rows(rows)

    output_dir = Path(args.output_dir)
    write_rows_csv(rows, output_dir / f"{args.split}_rows.csv")
    write_summary_plot(summary, output_dir / f"{args.split}_summary.png", title=f"Baseline {args.split} summary")
    (output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_training(args: argparse.Namespace) -> None:
    config = ExperimentConfig(train_timesteps=args.timesteps)
    run_name = args.run_name or f"{args.algorithm}_{args.timesteps}"
    run_dir = train_model(args.algorithm, config, PostProcessConfig(), run_name)
    print(run_dir)


def run_export_samples(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    artifacts = export_policy_samples(
        model_path=Path(args.model_path),
        split=args.split,
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        experiment_config=config,
        postprocess_config=PostProcessConfig(),
    )
    payload = [
        {
            "sample_id": artifact.sample_id,
            "output_path": str(artifact.output_path),
            "reward": artifact.reward,
            "steps": artifact.steps,
            "f1": artifact.f1,
            "boundary_f1": artifact.boundary_f1,
            "iou": artifact.iou,
        }
        for artifact in artifacts
    ]
    print(json.dumps(payload, indent=2))


def run_export_ods_binary(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    threshold_values = build_threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    selection_dataset = DatasetIndex(config.inputs_dir, args.selection_split)
    sweep_result = select_ods_threshold(selection_dataset, threshold_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "selection_split": args.selection_split,
        "target_splits": list(args.target_splits),
        "best_threshold": sweep_result.best_threshold,
        "best_summary": sweep_result.best_summary,
        "threshold_start": args.threshold_start,
        "threshold_stop": args.threshold_stop,
        "threshold_step": args.threshold_step,
    }
    (output_dir / "ods_threshold.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_ods_sweep_csv(sweep_result.sweep_entries, output_dir / "threshold_sweep.csv")
    write_summary_plot(sweep_result.best_summary, output_dir / "selection_summary.png", title=f"ODS selection {args.selection_split}")

    export_payload: dict[str, object] = {
        "best_threshold": sweep_result.best_threshold,
        "selection_split": args.selection_split,
        "splits": {},
    }
    for split in args.target_splits:
        dataset = DatasetIndex(config.inputs_dir, split)
        split_dir = output_dir / split
        rows, manifest_rows = export_ods_binary_maps(dataset, sweep_result.best_threshold, split_dir)
        summary = summarize_rows(rows)
        write_rows_csv(rows, split_dir / "rows.csv")
        write_export_manifest(manifest_rows, split_dir / "manifest.csv")
        write_summary_plot(summary, split_dir / "summary.png", title=f"ODS binary {split}")
        (split_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        export_payload["splits"][split] = {
            "count": len(rows),
            "summary": summary,
            "manifest": str(split_dir / "manifest.csv"),
        }

    print(json.dumps(export_payload, indent=2))


def run_evaluate_model(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    dataset = DatasetIndex(config.inputs_dir, args.split)
    rows = evaluate_policy_model(Path(args.model_path), dataset, PostProcessConfig())
    summary = summarize_policy_rows(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_policy_rows_csv(rows, output_dir / f"{args.split}_rows.csv")
    write_summary_plot(summary, output_dir / f"{args.split}_summary.png", title=f"Policy evaluation {args.split}")
    (output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_compare_runs(args: argparse.Namespace) -> None:
    dqn_run_dir = Path(args.dqn_run_dir)
    ppo_run_dir = Path(args.ppo_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dqn_training = read_metric_records_csv(dqn_run_dir / "csv_logs" / "training_metrics.csv")
    ppo_training = read_metric_records_csv(ppo_run_dir / "csv_logs" / "training_metrics.csv")
    merged_training = merge_metric_records(dqn_training, ppo_training)
    write_metric_records_csv(merged_training, output_dir / "training_metrics.csv")
    write_training_plots(merged_training, output_dir)

    dqn_eval = read_metric_records_csv(dqn_run_dir / "csv_logs" / "eval_compare_metrics.csv")
    ppo_eval = read_metric_records_csv(ppo_run_dir / "csv_logs" / "eval_compare_metrics.csv")
    merged_eval = merge_metric_records(dqn_eval, ppo_eval)
    write_metric_records_csv(merged_eval, output_dir / "eval_compare_metrics.csv")
    write_eval_plots(merged_eval, output_dir)

    payload = {
        "training_metrics": str(output_dir / "training_metrics.csv"),
        "eval_compare_metrics": str(output_dir / "eval_compare_metrics.csv"),
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "baseline":
        run_baseline(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "export-samples":
        run_export_samples(args)
    elif args.command == "export-ods-binary":
        run_export_ods_binary(args)
    elif args.command == "evaluate-model":
        run_evaluate_model(args)
    elif args.command == "compare-runs":
        run_compare_runs(args)


if __name__ == "__main__":
    main()