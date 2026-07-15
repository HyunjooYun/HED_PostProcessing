from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ParameterState
from .data import DatasetIndex
from .env import HEDEdgeEnv
from .inference import load_model
from .metrics import boundary_f1_score, iou_score, precision_recall_f1
from .postprocess import run_postprocess


@dataclass(slots=True)
class EvaluationRow:
    sample_id: str
    precision: float
    recall: float
    f1: float
    boundary_f1: float
    iou: float


@dataclass(slots=True)
class PolicyEvaluationRow:
    sample_id: str
    reward: float
    steps: int
    threshold_initial: float
    threshold_final: float
    threshold_mean: float
    threshold_std: float
    blur_sigma_initial: float
    blur_sigma_final: float
    blur_sigma_mean: float
    blur_sigma_std: float
    morphology_radius_initial: float
    morphology_radius_final: float
    morphology_radius_mean: float
    morphology_radius_std: float
    nms_strength_initial: float
    nms_strength_final: float
    nms_strength_mean: float
    nms_strength_std: float
    precision: float
    recall: float
    f1: float
    boundary_f1: float
    iou: float


def evaluate_fixed_threshold(dataset: DatasetIndex, params: ParameterState) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        processed = run_postprocess(sample.hed, params)
        precision, recall, f1 = precision_recall_f1(processed, sample.gt)
        _, _, boundary_f1 = boundary_f1_score(processed, sample.gt)
        iou = iou_score(processed, sample.gt)
        rows.append(EvaluationRow(sample.sample_id, precision, recall, f1, boundary_f1, iou))
    return rows


def summarize_rows(rows: list[EvaluationRow]) -> dict[str, float]:
    if not rows:
        return {"count": 0.0}
    metrics = np.array([[row.precision, row.recall, row.f1, row.boundary_f1, row.iou] for row in rows], dtype=np.float32)
    return {
        "count": float(len(rows)),
        "precision_mean": float(metrics[:, 0].mean()),
        "recall_mean": float(metrics[:, 1].mean()),
        "f1_mean": float(metrics[:, 2].mean()),
        "boundary_f1_mean": float(metrics[:, 3].mean()),
        "iou_mean": float(metrics[:, 4].mean()),
    }


def evaluate_policy_episodes(
    model,
    env: HEDEdgeEnv,
    num_episodes: int,
    progress_every: int | None = None,
    progress_label: str | None = None,
) -> list[PolicyEvaluationRow]:
    rows: list[PolicyEvaluationRow] = []
    env.restart_sequence()
    observation, reset_info = env.reset()
    for episode_index in range(num_episodes):
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        threshold_history = [float(reset_info["params"]["threshold"])]
        blur_sigma_history = [float(reset_info["params"]["blur_sigma"])]
        morphology_radius_history = [float(reset_info["params"]["morphology_radius"])]
        nms_strength_history = [float(reset_info["params"]["nms_strength"])]
        final_info: dict | None = None
        while not done and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, final_info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            threshold_history.append(float(final_info["params"]["threshold"]))
            blur_sigma_history.append(float(final_info["params"]["blur_sigma"]))
            morphology_radius_history.append(float(final_info["params"]["morphology_radius"]))
            nms_strength_history.append(float(final_info["params"]["nms_strength"]))

        if final_info is None:
            raise RuntimeError("Policy evaluation did not produce final step info")
        thresholds = np.asarray(threshold_history, dtype=np.float32)
        blur_sigmas = np.asarray(blur_sigma_history, dtype=np.float32)
        morphology_radii = np.asarray(morphology_radius_history, dtype=np.float32)
        nms_strengths = np.asarray(nms_strength_history, dtype=np.float32)
        metrics = final_info["metrics"]
        rows.append(
            PolicyEvaluationRow(
                sample_id=str(final_info["sample_id"]),
                reward=total_reward,
                steps=steps,
                threshold_initial=float(thresholds[0]),
                threshold_final=float(thresholds[-1]),
                threshold_mean=float(thresholds.mean()),
                threshold_std=float(thresholds.std()),
                blur_sigma_initial=float(blur_sigmas[0]),
                blur_sigma_final=float(blur_sigmas[-1]),
                blur_sigma_mean=float(blur_sigmas.mean()),
                blur_sigma_std=float(blur_sigmas.std()),
                morphology_radius_initial=float(morphology_radii[0]),
                morphology_radius_final=float(morphology_radii[-1]),
                morphology_radius_mean=float(morphology_radii.mean()),
                morphology_radius_std=float(morphology_radii.std()),
                nms_strength_initial=float(nms_strengths[0]),
                nms_strength_final=float(nms_strengths[-1]),
                nms_strength_mean=float(nms_strengths.mean()),
                nms_strength_std=float(nms_strengths.std()),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                f1=float(metrics["f1"]),
                boundary_f1=float(metrics["boundary_f1"]),
                iou=float(metrics["iou"]),
            )
        )
        if progress_every and ((episode_index + 1) % progress_every == 0 or (episode_index + 1) == num_episodes):
            label = progress_label or "policy-eval"
            sample_id = str(final_info["sample_id"])
            print(f"[{label}] {episode_index + 1}/{num_episodes} sample={sample_id} boundary_f1={float(metrics['boundary_f1']):.4f}")
        if episode_index + 1 < num_episodes:
            observation, reset_info = env.reset()
    return rows


def evaluate_policy_dataset(
    model,
    dataset: DatasetIndex,
    postprocess_config,
    progress_every: int | None = None,
    progress_label: str | None = None,
) -> list[PolicyEvaluationRow]:
    env = HEDEdgeEnv(dataset, postprocess_config, cycle_images=False)
    return evaluate_policy_episodes(
        model,
        env,
        len(dataset),
        progress_every=progress_every,
        progress_label=progress_label,
    )


def evaluate_policy_model(
    model_path,
    dataset: DatasetIndex,
    postprocess_config,
    progress_every: int | None = None,
) -> list[PolicyEvaluationRow]:
    model = load_model(model_path)
    progress_label = Path(model_path).stem
    return evaluate_policy_dataset(
        model,
        dataset,
        postprocess_config,
        progress_every=progress_every,
        progress_label=progress_label,
    )


def summarize_policy_rows(rows: list[PolicyEvaluationRow]) -> dict[str, float]:
    if not rows:
        return {"count": 0.0}
    metric_getters = {
        "reward": lambda row: row.reward,
        "steps": lambda row: row.steps,
        "threshold_mean": lambda row: row.threshold_mean,
        "threshold_std": lambda row: row.threshold_std,
        "blur_sigma_mean": lambda row: row.blur_sigma_mean,
        "blur_sigma_std": lambda row: row.blur_sigma_std,
        "morphology_radius_mean": lambda row: row.morphology_radius_mean,
        "morphology_radius_std": lambda row: row.morphology_radius_std,
        "nms_strength_mean": lambda row: row.nms_strength_mean,
        "nms_strength_std": lambda row: row.nms_strength_std,
        "precision": lambda row: row.precision,
        "recall": lambda row: row.recall,
        "f1": lambda row: row.f1,
        "boundary_f1": lambda row: row.boundary_f1,
        "iou": lambda row: row.iou,
    }
    summary: dict[str, float] = {"count": float(len(rows))}
    for metric_name, getter in metric_getters.items():
        values = np.asarray([float(getter(row)) for row in rows], dtype=np.float32)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std())
    summary["reward_max"] = float(np.asarray([row.reward for row in rows], dtype=np.float32).max())
    return summary