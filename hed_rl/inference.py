from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3 import DQN, PPO

from .config import ExperimentConfig, PostProcessConfig
from .data import DatasetIndex, InferenceDatasetIndex, Sample
from .env import HEDEdgeEnv, HEDInferenceEnv


@dataclass(slots=True)
class EpisodeArtifact:
    sample_id: str
    output_path: Path
    reward: float
    steps: int
    f1: float
    boundary_f1: float
    iou: float


@dataclass(slots=True)
class InferenceResult:
    sample_id: str
    steps: int
    processed: np.ndarray
    params_history: list[dict[str, float]]


def run_policy_inference(model, dataset: InferenceDatasetIndex, postprocess_config: PostProcessConfig, progress_every: int | None = None) -> list[InferenceResult]:
    """Run the policy for a fixed number of steps per image without touching GT."""
    env = HEDInferenceEnv(dataset, postprocess_config)
    results: list[InferenceResult] = []
    observation, info = env.reset()
    for episode_index in range(len(dataset)):
        params_history = [dict(info["params"])]
        truncated = False
        final_info = info
        steps = 0
        while not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, _, truncated, final_info = env.step(int(action))
            params_history.append(dict(final_info["params"]))
            steps += 1
        processed = env.processed_map
        if processed is None:
            raise RuntimeError("Inference environment did not expose a final edge map")
        results.append(
            InferenceResult(
                sample_id=str(final_info["sample_id"]),
                steps=steps,
                processed=processed.copy(),
                params_history=params_history,
            )
        )
        if progress_every and ((episode_index + 1) % progress_every == 0 or episode_index + 1 == len(dataset)):
            print(f"[infer] {episode_index + 1}/{len(dataset)} sample={final_info['sample_id']}")
        if episode_index + 1 < len(dataset):
            observation, info = env.reset()
    return results


def save_inference_outputs(results: list[InferenceResult], output_dir: Path) -> None:
    """Persist final edge maps and parameter trajectories. No metrics, no GT."""
    maps_dir = output_dir / "edge_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    trajectories: dict[str, list[dict[str, float]]] = {}
    for result in results:
        Image.fromarray((result.processed > 0.5).astype(np.uint8) * 255, mode="L").save(maps_dir / f"{result.sample_id}.png")
        trajectories[result.sample_id] = result.params_history
    (output_dir / "params_trajectory.json").write_text(json.dumps(trajectories), encoding="utf-8")
    (output_dir / "inference_manifest.json").write_text(
        json.dumps({"count": len(results), "sample_ids": [r.sample_id for r in results], "steps": sorted({r.steps for r in results})}, indent=2),
        encoding="utf-8",
    )


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    name = str(model_path).lower()
    if "ppo" in name:
        return PPO.load(str(model_path))
    return DQN.load(str(model_path))


def _save_episode_figure(sample: Sample, processed: np.ndarray, output_path: Path, metrics: dict[str, float], steps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(sample.image.astype(np.uint8))
    axes[0].set_title(f"Image\n{sample.sample_id}")
    axes[1].imshow(sample.hed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("HED")
    axes[2].imshow(sample.gt, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("GT")
    axes[3].imshow(processed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        "Policy Output\n"
        f"steps={steps}, f1={metrics['f1']:.3f}, bf1={metrics['boundary_f1']:.3f}, iou={metrics['iou']:.3f}"
    )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def export_policy_samples(
    model_path: Path,
    split: str,
    num_samples: int,
    output_dir: Path,
    experiment_config: ExperimentConfig,
    postprocess_config: PostProcessConfig,
) -> list[EpisodeArtifact]:
    model = load_model(model_path)
    dataset = DatasetIndex(experiment_config.inputs_dir, split)
    env = HEDEdgeEnv(dataset, postprocess_config, cycle_images=False)
    artifacts: list[EpisodeArtifact] = []

    observation, info = env.reset()
    for episode_index in range(num_samples):
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        final_info = info
        while not done and not truncated:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, final_info = env.step(int(action))
            total_reward += float(reward)
            steps += 1

        sample = env.sample
        processed = env.processed_map
        if sample is None or processed is None:
            raise RuntimeError("Environment did not expose sample outputs after episode execution")

        metrics = final_info["metrics"]
        output_path = output_dir / f"{episode_index + 1:02d}_{sample.sample_id}.png"
        _save_episode_figure(sample, processed, output_path, metrics, steps)
        artifacts.append(
            EpisodeArtifact(
                sample_id=sample.sample_id,
                output_path=output_path,
                reward=total_reward,
                steps=steps,
                f1=float(metrics["f1"]),
                boundary_f1=float(metrics["boundary_f1"]),
                iou=float(metrics["iou"]),
            )
        )

        if episode_index + 1 < num_samples:
            observation, info = env.reset()

    return artifacts