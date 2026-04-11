from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import ParameterState, PostProcessConfig
from .data import DatasetIndex, Sample
from .features import compute_observation_features
from .metrics import boundary_f1_score, iou_score, precision_recall_f1
from .postprocess import ACTIONS, apply_action, run_postprocess


@dataclass(slots=True)
class StepMetrics:
    precision: float
    recall: float
    f1: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    iou: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "boundary_precision": self.boundary_precision,
            "boundary_recall": self.boundary_recall,
            "boundary_f1": self.boundary_f1,
            "iou": self.iou,
        }


class HEDEdgeEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, dataset: DatasetIndex, config: PostProcessConfig, cycle_images: bool = True):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.cycle_images = cycle_images
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self._current_index = -1
        self._step_count = 0
        self._params = ParameterState().clip()
        self._sample: Sample | None = None
        self._processed = None

    def _next_sample(self) -> Sample:
        self._current_index += 1
        if self._current_index >= len(self.dataset):
            if not self.cycle_images:
                self._current_index = len(self.dataset) - 1
            else:
                self._current_index = 0
        return self.dataset[self._current_index]

    def _observe(self) -> np.ndarray:
        assert self._sample is not None
        assert self._processed is not None
        step_ratio = self._step_count / max(self.config.max_episode_steps, 1)
        return compute_observation_features(self._sample.hed, self._processed, self._params, step_ratio)

    def _evaluate(self) -> StepMetrics:
        assert self._sample is not None
        precision, recall, f1 = precision_recall_f1(self._processed, self._sample.gt)
        boundary_precision, boundary_recall, boundary_f1 = boundary_f1_score(
            self._processed,
            self._sample.gt,
            tolerance=self.config.boundary_tolerance,
        )
        iou = iou_score(self._processed, self._sample.gt)
        return StepMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            boundary_precision=boundary_precision,
            boundary_recall=boundary_recall,
            boundary_f1=boundary_f1,
            iou=iou,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._params = ParameterState().clip()
        self._sample = self._next_sample()
        self._processed = run_postprocess(self._sample.hed, self._params)
        metrics = self._evaluate()
        info = {
            "sample_id": self._sample.sample_id,
            "metrics": metrics.to_dict(),
            "params": self._params.to_dict(),
        }
        return self._observe(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._sample is not None
        action_name = ACTIONS[action]
        self._step_count += 1
        self._params = apply_action(self._params, action_name, self.config)
        self._processed = run_postprocess(self._sample.hed, self._params)
        metrics = self._evaluate()
        terminated = metrics.boundary_f1 >= 0.98
        truncated = self._step_count >= self.config.max_episode_steps
        info = {
            "sample_id": self._sample.sample_id,
            "action_name": action_name,
            "metrics": metrics.to_dict(),
            "params": self._params.to_dict(),
        }
        return self._observe(), metrics.boundary_f1, terminated, truncated, info

    def restart_sequence(self) -> None:
        self._current_index = -1

    @property
    def sample(self) -> Sample | None:
        return self._sample

    @property
    def processed_map(self) -> np.ndarray | None:
        return self._processed