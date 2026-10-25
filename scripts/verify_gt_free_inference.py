"""Verify that policy inference is independent of ground truth.

Checks:
  1. Inference succeeds on a copy of the inputs that has no GT_map directory.
  2. Outputs (edge maps and parameter trajectories) are identical with and without GT present.
  3. No file open / directory listing touching GT_map happens during inference (audit hook).
  4. GT is read only afterwards, by the offline evaluation (positive control for the audit hook),
     and optionally matches an existing evaluation CSV produced by the previous GT-coupled pipeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hed_rl.config import PostProcessConfig
from hed_rl.data import InferenceDatasetIndex
from hed_rl.evaluation import evaluate_saved_outputs
from hed_rl.inference import InferenceResult, load_model, run_policy_inference, save_inference_outputs

GT_ACCESS_LOG: list[str] = []
_AUDIT_ACTIVE = False
_AUDITED_EVENTS = {"open", "os.listdir", "os.scandir", "os.stat"}


def _audit_hook(event: str, args: tuple) -> None:
    if not _AUDIT_ACTIVE or event not in _AUDITED_EVENTS or not args:
        return
    target = args[0]
    if isinstance(target, (str, bytes, Path)) and "gt_map" in str(target).lower():
        GT_ACCESS_LOG.append(f"{event}: {target}")


sys.addaudithook(_audit_hook)


def _set_audit(active: bool) -> None:
    global _AUDIT_ACTIVE
    _AUDIT_ACTIVE = active


def _copy_inputs_without_gt(inputs_dir: Path, split: str, target: Path, limit: int | None) -> int:
    image_paths = sorted((inputs_dir / "images" / split).glob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]
    (target / "images" / split).mkdir(parents=True)
    (target / "HED_map" / split).mkdir(parents=True)
    for image_path in image_paths:
        shutil.copy2(image_path, target / "images" / split / image_path.name)
        hed_path = inputs_dir / "HED_map" / split / f"{image_path.stem}_edge.png"
        shutil.copy2(hed_path, target / "HED_map" / split / hed_path.name)
    return len(image_paths)


def _limit(dataset: InferenceDatasetIndex, limit: int | None) -> InferenceDatasetIndex:
    if limit:
        dataset.samples = dataset.samples[:limit]
    return dataset


def _compare(results_a: list[InferenceResult], results_b: list[InferenceResult]) -> list[str]:
    problems: list[str] = []
    if [r.sample_id for r in results_a] != [r.sample_id for r in results_b]:
        return ["sample order/ids differ"]
    for a, b in zip(results_a, results_b):
        if not np.array_equal(a.processed, b.processed):
            problems.append(f"{a.sample_id}: edge map differs")
        if a.params_history != b.params_history:
            problems.append(f"{a.sample_id}: parameter trajectory differs")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--inputs-dir", default=str(PROJECT_ROOT / "inputs"))
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N images (smoke test)")
    parser.add_argument("--reference-rows", default=None, help="Existing *_rows.csv from the previous pipeline to compare against")
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    inputs_dir = Path(args.inputs_dir)
    config = PostProcessConfig()
    model = load_model(Path(args.model_path))
    report: dict[str, object] = {"model_path": args.model_path, "split": args.split}
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        no_gt_inputs = tmp_path / "inputs_without_gt"
        count = _copy_inputs_without_gt(inputs_dir, args.split, no_gt_inputs, args.limit)
        assert not (no_gt_inputs / "GT_map").exists()

        # Check 1: inference on inputs that contain no GT at all.
        results_without_gt = run_policy_inference(model, InferenceDatasetIndex(no_gt_inputs, args.split), config)
        report["without_gt_count"] = len(results_without_gt)
        if len(results_without_gt) != count:
            failures.append(f"[1] expected {count} results without GT, got {len(results_without_gt)}")
        if any(r.steps != config.max_episode_steps for r in results_without_gt):
            failures.append("[1] some episodes did not run exactly max_episode_steps")

        # Checks 2 and 3: inference with GT present on disk, audited for any GT_map access.
        GT_ACCESS_LOG.clear()
        _set_audit(True)
        try:
            dataset = _limit(InferenceDatasetIndex(inputs_dir, args.split), args.limit)
            results_with_gt = run_policy_inference(model, dataset, config)
        finally:
            _set_audit(False)
        report["gt_accesses_during_inference"] = list(GT_ACCESS_LOG)
        if GT_ACCESS_LOG:
            failures.append(f"[3] GT_map accessed during inference ({len(GT_ACCESS_LOG)} events), e.g. {GT_ACCESS_LOG[0]}")

        problems = _compare(results_without_gt, results_with_gt)
        report["output_mismatches"] = problems
        if problems:
            failures.append(f"[2] outputs differ with/without GT ({len(problems)} mismatches), e.g. {problems[0]}")

        # Check 4: GT is read only by the offline evaluation (also validates the audit hook).
        inference_dir = tmp_path / "inference_outputs"
        save_inference_outputs(results_without_gt, inference_dir)
        GT_ACCESS_LOG.clear()
        _set_audit(True)
        try:
            rows = evaluate_saved_outputs(inference_dir, inputs_dir, args.split)
        finally:
            _set_audit(False)
        report["gt_accesses_during_offline_eval"] = len(GT_ACCESS_LOG)
        if not GT_ACCESS_LOG:
            failures.append("[4] audit hook saw no GT access during offline evaluation; the hook is not trustworthy")
        report["offline_eval_summary"] = {
            "count": len(rows),
            "boundary_f1_mean": float(np.mean([r.boundary_f1 for r in rows])),
            "f1_mean": float(np.mean([r.f1 for r in rows])),
            "iou_mean": float(np.mean([r.iou for r in rows])),
        }

        if args.reference_rows:
            reference = pd.read_csv(args.reference_rows, dtype={"sample_id": str}).set_index("sample_id")
            reference = reference.loc[[row.sample_id for row in rows]]
            candidates: dict[str, np.ndarray] = {
                column: np.asarray([getattr(row, column) for row in rows], dtype=np.float64)
                for column in ("f1", "boundary_f1", "iou", "precision", "recall", "steps")
            }
            for name in ("threshold", "blur_sigma", "morphology_radius", "nms_strength"):
                candidates[f"{name}_final"] = np.asarray([row.params_final[name] for row in rows], dtype=np.float64)
                candidates[f"{name}_mean"] = np.asarray([row.params_mean[name] for row in rows], dtype=np.float64)
            compared = [column for column in candidates if column in reference.columns]
            max_diff = 0.0
            for column in compared:
                diff = float(np.abs(candidates[column] - reference[column].to_numpy(dtype=np.float64)).max())
                max_diff = max(max_diff, diff)
            report["reference_columns_compared"] = compared
            report["max_abs_diff_vs_reference"] = max_diff
            if max_diff > 1e-5:
                failures.append(f"[ref] metrics differ from reference rows (max abs diff {max_diff:.3e})")

    report["passed"] = not failures
    report["failures"] = failures
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
