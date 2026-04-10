from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import ExperimentConfig, PostProcessConfig
from .data import DatasetIndex
from .env import HEDEdgeEnv
from .reporting import write_eval_plots, write_metric_records_csv, write_stability_plots, write_training_plots
from .tracking import DQNStabilityCallback, EvaluationTraceCallback, PPOStabilityCallback, TrainingMetricsCallback


def build_env(inputs_dir: Path, split: str, postprocess_config: PostProcessConfig, cycle_images: bool) -> HEDEdgeEnv:
    dataset = DatasetIndex(inputs_dir, split)
    return HEDEdgeEnv(dataset, postprocess_config, cycle_images=cycle_images)


def train_model(algorithm: str, config: ExperimentConfig, postprocess_config: PostProcessConfig, run_name: str):
    config.ensure_directories()
    train_env = DummyVecEnv([
        lambda: Monitor(build_env(config.inputs_dir, "train", postprocess_config, config.cycle_images))
    ])
    eval_env = DummyVecEnv([
        lambda: Monitor(build_env(config.inputs_dir, "val", postprocess_config, True))
    ])
    run_dir = config.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = run_dir / "csv_logs"
    plot_dir = run_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if algorithm.lower() == "dqn":
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_starts=1_000,
            buffer_size=50_000,
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=config.seed,
        )
    elif algorithm.lower() == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            n_steps=512,
            batch_size=128,
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    training_callback = TrainingMetricsCallback(
        algorithm.lower(),
        log_interval=1_000,
        output_path=csv_dir / "training_metrics.csv",
    )
    evaluation_callback = EvaluationTraceCallback(
        algorithm.lower(),
        config,
        postprocess_config,
        run_dir,
        log_interval=1_000,
        output_path=csv_dir / "eval_compare_metrics.csv",
    )
    callbacks = [training_callback, evaluation_callback]
    if algorithm.lower() == "dqn":
        stability_callback = DQNStabilityCallback(log_interval=1_000, output_path=csv_dir / "dqn_stability_metrics.csv")
    else:
        stability_callback = PPOStabilityCallback(log_interval=1_000, output_path=csv_dir / "ppo_stability_metrics.csv")
    callbacks.append(stability_callback)

    model.learn(total_timesteps=config.train_timesteps, callback=CallbackList(callbacks), progress_bar=False)
    model.save(str(run_dir / f"{algorithm.lower()}_final"))

    write_metric_records_csv(training_callback.records, csv_dir / "training_metrics.csv")
    write_training_plots(training_callback.records, plot_dir)
    write_metric_records_csv(evaluation_callback.records, csv_dir / "eval_compare_metrics.csv")
    write_eval_plots(evaluation_callback.records, plot_dir)

    if algorithm.lower() == "dqn":
        write_metric_records_csv(stability_callback.records, csv_dir / "dqn_stability_metrics.csv")
        write_stability_plots("dqn", stability_callback.records, plot_dir)
    else:
        write_metric_records_csv(stability_callback.records, csv_dir / "ppo_stability_metrics.csv")
        write_stability_plots("ppo", stability_callback.records, plot_dir)
    return run_dir