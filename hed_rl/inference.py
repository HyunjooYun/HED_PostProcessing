from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO

from .config import ExperimentConfig, PostProcessConfig
from .data import DatasetIndex, Sample
from .env import HEDEdgeEnv


@dataclass(slots=True)
class EpisodeArtifact:
    sample_id: str
    output_path: Path
    reward: float
    steps: int
    f1: float
    boundary_f1: float
    iou: float


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