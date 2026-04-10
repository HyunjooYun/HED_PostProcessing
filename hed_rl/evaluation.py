from __future__ import annotations

from dataclasses import dataclass

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


def evaluate_policy_episodes(model, env: HEDEdgeEnv, num_episodes: int) -> list[PolicyEvaluationRow]:
    rows: list[PolicyEvaluationRow] = []
    env.restart_sequence()
    observation, _ = env.reset()
    for episode_index in range(num_episodes):
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        final_info: dict | None = None
        while not done and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, final_info = env.step(int(action))
            total_reward += float(reward)
            steps += 1

        if final_info is None:
            raise RuntimeError("Policy evaluation did not produce final step info")
        metrics = final_info["metrics"]
        rows.append(
            PolicyEvaluationRow(
                sample_id=str(final_info["sample_id"]),
                reward=total_reward,
                steps=steps,
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                f1=float(metrics["f1"]),
                boundary_f1=float(metrics["boundary_f1"]),
                iou=float(metrics["iou"]),
            )
        )
        if episode_index + 1 < num_episodes:
            observation, _ = env.reset()
    return rows


def evaluate_policy_dataset(model, dataset: DatasetIndex, postprocess_config) -> list[PolicyEvaluationRow]:
    env = HEDEdgeEnv(dataset, postprocess_config, cycle_images=False)
    return evaluate_policy_episodes(model, env, len(dataset))


def evaluate_policy_model(model_path, dataset: DatasetIndex, postprocess_config) -> list[PolicyEvaluationRow]:
    model = load_model(model_path)
    return evaluate_policy_dataset(model, dataset, postprocess_config)


def summarize_policy_rows(rows: list[PolicyEvaluationRow]) -> dict[str, float]:
    if not rows:
        return {"count": 0.0}
    metrics = np.array(
        [[row.reward, row.steps, row.precision, row.recall, row.f1, row.boundary_f1, row.iou] for row in rows],
        dtype=np.float32,
    )
    return {
        "count": float(len(rows)),
        "reward_mean": float(metrics[:, 0].mean()),
        "reward_std": float(metrics[:, 0].std()),
        "reward_max": float(metrics[:, 0].max()),
        "steps_mean": float(metrics[:, 1].mean()),
        "precision_mean": float(metrics[:, 2].mean()),
        "recall_mean": float(metrics[:, 3].mean()),
        "f1_mean": float(metrics[:, 4].mean()),
        "boundary_f1_mean": float(metrics[:, 5].mean()),
        "iou_mean": float(metrics[:, 6].mean()),
    }