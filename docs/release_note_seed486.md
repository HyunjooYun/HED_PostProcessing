# Release Note: Seed 486 Experiment Update

## Summary

This update adds DQN `q_max` tracking, reruns the main DQN and PPO experiments with seed `486`, and refreshes the comparison artifacts used for reporting and paper drafting.

## Code changes

- Added `q_max` logging to the DQN stability callback.
- Added `q_max` to the exported DQN stability plot set.

Relevant files:

- `hed_rl/tracking.py`
- `hed_rl/reporting.py`

## New experiment artifacts

- DQN seed-486 run: `runs/dqn_seed486_sqrt8/`
- PPO seed-486 run: `runs/ppo_seed486_sqrt8/`
- Shared comparison plots: `runs/seed486_compare/`
- Updated all-seed summary: `runs/all_seed_test_summary.csv`
- Paper-ready summary: `runs/paper_seed486_summary.md`

## Key results

| method | reward_mean | binary_f1_mean | boundary_f1_mean | iou_mean |
| --- | ---: | ---: | ---: | ---: |
| HED baseline | 0.0000 | 0.4117 | 0.7551 | 0.2614 |
| DQN seed 486 | 22.5252 | 0.4088 | 0.7515 | 0.2591 |
| PPO seed 486 | 22.3334 | 0.3901 | 0.7262 | 0.2445 |

## Interpretation

- DQN remains close to the HED baseline on binary F1 and IoU.
- PPO underperforms both HED and DQN on the seed-486 test split.
- DQN improves over PPO on boundary F1 for this seed, but does not exceed the HED baseline.

## Aggregate over seeds

Using seeds `24`, `42`, `123`, `486`, and `999`:

- DQN boundary F1 mean +/- std: `0.7508 +/- 0.0010`
- PPO boundary F1 mean +/- std: `0.7449 +/- 0.0096`

## Recommended figures

- DQN Q-max stability: `runs/dqn_seed486_sqrt8/plots/dqn_stability_q_max.png`
- Shared DQN/PPO validation reward: `runs/seed486_compare/eval_compare_reward_mean.png`
- HED vs DQN vs PPO boundary F1: `runs/seed486_compare/test_compare_boundary_f1.png`
- HED vs DQN vs PPO binary F1: `runs/seed486_compare/test_compare_binary_f1.png`
- HED vs DQN vs PPO IoU: `runs/seed486_compare/test_compare_iou.png`