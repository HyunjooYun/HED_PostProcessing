# HED Post-Processing RL

HED edge map post-processing parameter optimization scaffold based on the project plan.

## Implemented scope

- Paired dataset loader for `inputs/images`, `inputs/HED_map`, `inputs/GT_map`
- Post-processing pipeline with `threshold`, `blur_sigma`, `morphology_radius`, `nms_strength`
- Gymnasium environment for image-wise RL optimization
- Evaluation metrics: pixel-wise precision/recall/F1, IoU, boundary-tolerant F1
- Stable-Baselines3 training entry points for DQN and PPO
- CSV, JSON, PNG report export for baseline evaluation

## Project structure

- `main.py`: CLI entry point
- `hed_rl/data.py`: dataset discovery and loading
- `hed_rl/postprocess.py`: post-processing pipeline and action mapping
- `hed_rl/metrics.py`: F1, IoU, boundary-aware F1
- `hed_rl/env.py`: Gymnasium environment
- `hed_rl/training.py`: DQN and PPO training helpers
- `hed_rl/evaluation.py`: fixed-parameter evaluation
- `hed_rl/reporting.py`: CSV and plot export

## Current design choices

- Both DQN and PPO use the same discrete action space for fair comparison.
- Reward uses boundary-aware F1 as a practical approximation of the BPR-driven objective from the plan.
- One episode processes one image with up to 30 steps.
- Dataset traversal cycles through images split by split.

## Environment

The current virtual environment already works with the required packages in `requirements.txt`.

## Documentation

- Korean development guide: `docs/development_guide_ko.md`

## Commands

Run a fixed baseline on the validation set:

```powershell
& ".venv/Scripts/python.exe" .\main.py baseline --split val --output-dir runs\baseline_val
```

Train DQN:

```powershell
& ".venv/Scripts/python.exe" .\main.py train --algorithm dqn --timesteps 200000 --run-name dqn_main
```

Train PPO:

```powershell
& ".venv/Scripts/python.exe" .\main.py train --algorithm ppo --timesteps 200000 --run-name ppo_main
```

Export 10 sample images from a trained model:

```powershell
& ".venv/Scripts/python.exe" .\main.py export-samples --model-path runs\dqn_main\dqn_final.zip --split test --num-samples 10 --output-dir runs\dqn_main\samples
```

Generate shared DQN/PPO comparison graphs for the same metrics:

```powershell
& ".venv/Scripts/python.exe" .\main.py compare-runs --dqn-run-dir runs\dqn_sqrt8 --ppo-run-dir runs\ppo_sqrt8 --output-dir runs\sqrt8_compare
```

This writes merged CSV logs plus one-figure comparison plots such as `training_episode_reward_mean.png` and `eval_compare_reward_mean.png` into the output directory.

Launch TensorBoard:

```powershell
& ".venv/Scripts/python.exe" -m tensorboard.main --logdir runs
```

## Verified smoke tests

- Baseline evaluation on `val`
- DQN training entry point with short smoke run
- PPO training entry point with short smoke run

## Next extensions

- Replace the boundary-aware F1 approximation with a stricter BPR implementation
- Add per-step CSV logging during RL training
- Save qualitative comparison images for HED, GT, baseline, DQN, and PPO outputs
- Add held-out evaluation script that loads trained checkpoints and exports split-wise summaries
