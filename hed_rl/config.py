from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ParameterState:
    threshold: float = 0.5
    blur_sigma: float = 1.0
    morphology_radius: int = 1
    nms_strength: float = 0.5

    def clip(self) -> "ParameterState":
        self.threshold = min(max(self.threshold, 0.05), 1.0)
        self.blur_sigma = min(max(self.blur_sigma, 0.2), 3.0)
        self.morphology_radius = int(min(max(self.morphology_radius, 1), 7))
        self.nms_strength = min(max(self.nms_strength, 0.2), 1.0)
        return self

    def to_vector(self) -> list[float]:
        return [
            self.threshold,
            self.blur_sigma,
            float(self.morphology_radius),
            self.nms_strength,
        ]


@dataclass(slots=True)
class PostProcessConfig:
    threshold_step: float = 0.05
    blur_step: float = 0.2
    morphology_step: int = 1
    nms_step: float = 0.1
    boundary_tolerance: float = math.sqrt(8.0)
    max_episode_steps: int = 30


@dataclass(slots=True)
class ExperimentConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    inputs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "inputs")
    runs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "runs")
    seed: int = 42
    train_timesteps: int = 200_000
    eval_episodes: int = 200
    cycle_images: bool = True
    image_splits: tuple[str, ...] = ("train", "val", "test")

    def ensure_directories(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)