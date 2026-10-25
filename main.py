from __future__ import annotations

import argparse
import json
from pathlib import Path

from hed_rl.config import ExperimentConfig, ParameterState, PostProcessConfig
from hed_rl.data import DatasetIndex, InferenceDatasetIndex
from hed_rl.evaluation import evaluate_fixed_threshold, evaluate_saved_outputs, summarize_offline_rows, summarize_rows
from hed_rl.inference import export_policy_samples, load_model, run_policy_inference, save_inference_outputs
from hed_rl.ois import export_ois_binary_maps, summarize_ois_thresholds, write_ois_thresholds_csv
from hed_rl.ods import build_threshold_grid, export_ods_binary_maps, select_ods_threshold, write_export_manifest, write_ods_sweep_csv
from hed_rl.otsu import export_otsu_binary_maps, summarize_otsu_thresholds, write_otsu_thresholds_csv
from hed_rl.reporting import read_metric_records_csv, write_compare_metric_plot, write_eval_plots, write_metric_records_csv, write_offline_rows_csv, write_rows_csv, write_summary_plot, write_training_plots, merge_metric_records
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
    train.add_argument("--seed", type=int, default=42)

    evaluate = subparsers.add_parser(
        "evaluate-model",
        help="GT-free inference followed by offline GT evaluation of a trained DQN or PPO model",
    )
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument("--output-dir", required=True)

    infer = subparsers.add_parser("infer-model", help="GT-free inference: saves final edge maps and parameter trajectories")
    infer.add_argument("--model-path", required=True)
    infer.add_argument("--split", default="test", choices=["train", "val", "test"])
    infer.add_argument("--output-dir", required=True)

    eval_outputs = subparsers.add_parser("evaluate-outputs", help="Offline GT evaluation of saved inference outputs")
    eval_outputs.add_argument("--inference-dir", required=True)
    eval_outputs.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_outputs.add_argument("--output-dir", default=None)

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

    ois = subparsers.add_parser("export-ois-binary", help="Select a per-image optimal threshold with GT and export OIS binary edge maps")
    ois.add_argument("--target-splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ois.add_argument("--threshold-start", type=float, default=0.05)
    ois.add_argument("--threshold-stop", type=float, default=1.0)
    ois.add_argument("--threshold-step", type=float, default=0.05)
    ois.add_argument("--output-dir", default="runs/ois_binary_maps")

    otsu = subparsers.add_parser("export-otsu-binary", help="Apply per-image Otsu thresholding and export binary edge maps")
    otsu.add_argument("--target-splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    otsu.add_argument("--output-dir", default="runs/otsu_binary_maps")

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
    config = ExperimentConfig(train_timesteps=args.timesteps, seed=args.seed)
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


def run_export_ois_binary(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    threshold_values = build_threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_payload: dict[str, object] = {
        "target_splits": list(args.target_splits),
        "threshold_start": args.threshold_start,
        "threshold_stop": args.threshold_stop,
        "threshold_step": args.threshold_step,
        "splits": {},
    }

    for split in args.target_splits:
        dataset = DatasetIndex(config.inputs_dir, split)
        split_dir = output_dir / split
        export_result = export_ois_binary_maps(dataset, threshold_values, split_dir)
        threshold_summary = summarize_ois_thresholds(export_result.sample_results)

        write_rows_csv(export_result.rows, split_dir / "rows.csv")
        write_ois_thresholds_csv(export_result.sample_results, split_dir / "ois_thresholds.csv")
        write_export_manifest(export_result.manifest_rows, split_dir / "manifest.csv")
        write_summary_plot(export_result.summary, split_dir / "summary.png", title=f"OIS binary {split}")
        (split_dir / "summary.json").write_text(json.dumps(export_result.summary, indent=2), encoding="utf-8")
        (split_dir / "threshold_summary.json").write_text(json.dumps(threshold_summary, indent=2), encoding="utf-8")

        export_payload["splits"][split] = {
            "count": len(export_result.rows),
            "summary": export_result.summary,
            "threshold_summary": threshold_summary,
            "manifest": str(split_dir / "manifest.csv"),
            "ois_thresholds": str(split_dir / "ois_thresholds.csv"),
        }

    print(json.dumps(export_payload, indent=2))


def run_export_otsu_binary(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_payload: dict[str, object] = {
        "target_splits": list(args.target_splits),
        "splits": {},
    }

    for split in args.target_splits:
        dataset = DatasetIndex(config.inputs_dir, split)
        split_dir = output_dir / split
        export_result = export_otsu_binary_maps(dataset, split_dir)
        threshold_summary = summarize_otsu_thresholds(export_result.sample_results)

        write_rows_csv(export_result.rows, split_dir / "rows.csv")
        write_otsu_thresholds_csv(export_result.sample_results, split_dir / "otsu_thresholds.csv")
        write_export_manifest(export_result.manifest_rows, split_dir / "manifest.csv")
        write_summary_plot(export_result.summary, split_dir / "summary.png", title=f"Otsu binary {split}")
        (split_dir / "summary.json").write_text(json.dumps(export_result.summary, indent=2), encoding="utf-8")
        (split_dir / "threshold_summary.json").write_text(json.dumps(threshold_summary, indent=2), encoding="utf-8")

        export_payload["splits"][split] = {
            "count": len(export_result.rows),
            "summary": export_result.summary,
            "threshold_summary": threshold_summary,
            "manifest": str(split_dir / "manifest.csv"),
            "otsu_thresholds": str(split_dir / "otsu_thresholds.csv"),
        }

    print(json.dumps(export_payload, indent=2))


def run_infer_model(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    dataset = InferenceDatasetIndex(config.inputs_dir, args.split)
    model = load_model(Path(args.model_path))
    results = run_policy_inference(model, dataset, PostProcessConfig(), progress_every=10)
    save_inference_outputs(results, Path(args.output_dir))
    print(f"Saved {len(results)} GT-free outputs to {args.output_dir}")


def run_evaluate_outputs(args: argparse.Namespace) -> None:
    config = ExperimentConfig()
    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir) if args.output_dir else inference_dir
    rows = evaluate_saved_outputs(inference_dir, config.inputs_dir, args.split)
    summary = summarize_offline_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_offline_rows_csv(rows, output_dir / f"{args.split}_rows.csv")
    write_summary_plot(summary, output_dir / f"{args.split}_summary.png", title=f"Policy evaluation {args.split}")
    (output_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_evaluate_model(args: argparse.Namespace) -> None:
    run_infer_model(args)
    args.inference_dir = args.output_dir
    run_evaluate_outputs(args)


def run_compare_runs(args: argparse.Namespace) -> None:
    dqn_run_dir = Path(args.dqn_run_dir)
    ppo_run_dir = Path(args.ppo_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dqn_training = read_metric_records_csv(dqn_run_dir / "csv_logs" / "training_metrics.csv")
    ppo_training = read_metric_records_csv(ppo_run_dir / "csv_logs" / "training_metrics.csv")
    merged_training = merge_metric_records(dqn_training, ppo_training)
    write_metric_records_csv(merged_training, output_dir / "training_metrics.csv")
    training_plot_paths = write_training_plots(merged_training, output_dir)

    dqn_eval = read_metric_records_csv(dqn_run_dir / "csv_logs" / "eval_compare_metrics.csv")
    ppo_eval = read_metric_records_csv(ppo_run_dir / "csv_logs" / "eval_compare_metrics.csv")
    merged_eval = merge_metric_records(dqn_eval, ppo_eval)
    write_metric_records_csv(merged_eval, output_dir / "eval_compare_metrics.csv")
    eval_plot_paths = write_eval_plots(merged_eval, output_dir)

    payload = {
        "dqn_run_dir": str(dqn_run_dir),
        "ppo_run_dir": str(ppo_run_dir),
        "training_metrics": str(output_dir / "training_metrics.csv"),
        "eval_compare_metrics": str(output_dir / "eval_compare_metrics.csv"),
        "training_plots": [str(path) for path in training_plot_paths],
        "eval_plots": [str(path) for path in eval_plot_paths],
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
    elif args.command == "export-ois-binary":
        run_export_ois_binary(args)
    elif args.command == "export-otsu-binary":
        run_export_otsu_binary(args)
    elif args.command == "evaluate-model":
        run_evaluate_model(args)
    elif args.command == "infer-model":
        run_infer_model(args)
    elif args.command == "evaluate-outputs":
        run_evaluate_outputs(args)
    elif args.command == "compare-runs":
        run_compare_runs(args)


if __name__ == "__main__":
    main()