from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from .config import ExperimentConfig, PostProcessConfig
from .data import DatasetIndex
from .env import HEDEdgeEnv
from .evaluation import evaluate_policy_episodes, summarize_policy_rows
from .reporting import write_metric_records_csv


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, algorithm: str, log_interval: int = 1_000, output_path: Path | None = None):
        super().__init__()
        self.algorithm = algorithm
        self.log_interval = log_interval
        self.output_path = output_path
        self.next_log_step = log_interval
        self.records: list[dict[str, float | str]] = []

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_log_step:
            rewards = [float(item["r"]) for item in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
            lengths = [float(item["l"]) for item in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
            self.records.append(
                {
                    "timestep": float(self.next_log_step),
                    "algorithm": self.algorithm,
                    "reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
                    "reward_std": float(np.std(rewards)) if rewards else float("nan"),
                    "length_mean": float(np.mean(lengths)) if lengths else float("nan"),
                    "length_std": float(np.std(lengths)) if lengths else float("nan"),
                    "episode_reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
                    "episode_reward_std": float(np.std(rewards)) if rewards else float("nan"),
                    "episode_length_mean": float(np.mean(lengths)) if lengths else float("nan"),
                    "episode_length_std": float(np.std(lengths)) if lengths else float("nan"),
                }
            )
            if self.output_path is not None:
                write_metric_records_csv(self.records, self.output_path)
            self.next_log_step += self.log_interval
        return True


class DQNStabilityCallback(BaseCallback):
    def __init__(self, log_interval: int = 1_000, output_path: Path | None = None):
        super().__init__()
        self.log_interval = log_interval
        self.output_path = output_path
        self.next_log_step = log_interval
        self.records: list[dict[str, float | str]] = []

    def _on_step(self) -> bool:
        model = self.model
        if not isinstance(model, DQN):
            return True

        while self.num_timesteps >= self.next_log_step:
            record = {
                "timestep": float(self.next_log_step),
                "algorithm": "dqn",
                "q_mean": float("nan"),
                "q_std": float("nan"),
                "td_error_mean": float("nan"),
                "td_error_std": float("nan"),
                "td_error_abs_mean": float("nan"),
                "exploration_rate": float(getattr(model, "exploration_rate", float("nan"))),
            }

            replay_buffer = getattr(model, "replay_buffer", None)
            learning_starts = int(getattr(model, "learning_starts", 0))
            buffer_size = replay_buffer.size() if replay_buffer is not None else 0
            if replay_buffer is not None and self.num_timesteps >= learning_starts and buffer_size > 1:
                batch_size = min(int(getattr(model, "batch_size", 32)), buffer_size)
                sample = replay_buffer.sample(batch_size, env=model._vec_normalize_env)
                with th.no_grad():
                    q_values_all = model.q_net(sample.observations)
                    current_q = q_values_all.gather(1, sample.actions.long())
                    next_q = model.q_net_target(sample.next_observations).max(dim=1, keepdim=True)[0]
                    target_q = sample.rewards + (1 - sample.dones) * model.gamma * next_q
                    td_error = target_q - current_q
                record.update(
                    {
                        "q_mean": float(q_values_all.mean().item()),
                        "q_std": float(q_values_all.std(unbiased=False).item()),
                        "td_error_mean": float(td_error.mean().item()),
                        "td_error_std": float(td_error.std(unbiased=False).item()),
                        "td_error_abs_mean": float(td_error.abs().mean().item()),
                    }
                )

            self.records.append(record)
            if self.output_path is not None:
                write_metric_records_csv(self.records, self.output_path)
            self.next_log_step += self.log_interval
        return True


class PPOStabilityCallback(BaseCallback):
    def __init__(self, log_interval: int = 1_000, output_path: Path | None = None):
        super().__init__()
        self.log_interval = log_interval
        self.output_path = output_path
        self.next_log_step = log_interval
        self.records: list[dict[str, float | str]] = []

    def _on_step(self) -> bool:
        model = self.model
        if not isinstance(model, PPO):
            return True

        while self.num_timesteps >= self.next_log_step:
            logger_values = getattr(model.logger, "name_to_value", {})
            self.records.append(
                {
                    "timestep": float(self.next_log_step),
                    "algorithm": "ppo",
                    "explained_variance": _float_or_nan(logger_values.get("train/explained_variance")),
                    "approx_kl": _float_or_nan(logger_values.get("train/approx_kl")),
                    "clip_fraction": _float_or_nan(logger_values.get("train/clip_fraction")),
                    "value_loss": _float_or_nan(logger_values.get("train/value_loss")),
                    "entropy_loss": _float_or_nan(logger_values.get("train/entropy_loss")),
                }
            )
            if self.output_path is not None:
                write_metric_records_csv(self.records, self.output_path)
            self.next_log_step += self.log_interval
        return True


class EvaluationTraceCallback(BaseCallback):
    def __init__(
        self,
        algorithm: str,
        config: ExperimentConfig,
        postprocess_config: PostProcessConfig,
        run_dir: Path,
        log_interval: int = 1_000,
        output_path: Path | None = None,
    ):
        super().__init__()
        self.algorithm = algorithm
        self.config = config
        self.postprocess_config = postprocess_config
        self.run_dir = run_dir
        self.log_interval = log_interval
        self.output_path = output_path
        self.next_log_step = log_interval
        self.records: list[dict[str, float | str]] = []
        self.best_reward_mean = float("-inf")
        self.eval_dataset = DatasetIndex(config.inputs_dir, "val")
        self.eval_env = HEDEdgeEnv(self.eval_dataset, postprocess_config, cycle_images=False)

    def _evaluate_model(self) -> dict[str, float]:
        episodes = min(self.config.eval_episodes, len(self.eval_dataset))
        rows = evaluate_policy_episodes(self.model, self.eval_env, episodes)
        return summarize_policy_rows(rows)

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_log_step:
            summary = self._evaluate_model()
            reward_mean = float(summary.get("reward_mean", float("nan")))
            reward_std = float(summary.get("reward_std", float("nan")))
            reward_max = float(summary.get("reward_max", float("nan")))
            self.records.append(
                {
                    "timestep": float(self.next_log_step),
                    "algorithm": self.algorithm,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "reward_max": reward_max,
                }
            )
            if self.output_path is not None:
                write_metric_records_csv(self.records, self.output_path)

            if reward_mean > self.best_reward_mean:
                self.best_reward_mean = reward_mean
                best_model_dir = self.run_dir / "best_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(str(best_model_dir / "best_model"))

            self.next_log_step += self.log_interval
        return True
