# Seed 486 Paper-ready Summary

## Test Performance Table

| method | reward_mean | binary_f1_mean | boundary_f1_mean | iou_mean |
| --- | ---: | ---: | ---: | ---: |
| HED baseline | 0.0000 | 0.4117 | 0.7551 | 0.2614 |
| DQN seed 486 | 22.5252 | 0.4088 | 0.7515 | 0.2591 |
| PPO seed 486 | 22.3334 | 0.3901 | 0.7262 | 0.2445 |

## Suggested Figure Captions

1. Shared training curves: Compare DQN and PPO learning behavior on seed 486 using episode reward, episode length, and validation reward. See runs/seed486_compare/training_episode_reward_mean.png, runs/seed486_compare/training_episode_length_mean.png, and runs/seed486_compare/eval_compare_reward_mean.png.
2. DQN stability with Q-max: Show DQN internal value-scale dynamics over training, including q_max, q_mean, TD-error statistics, and exploration rate. See runs/dqn_seed486_sqrt8/plots/dqn_stability_q_max.png and related DQN stability plots.
3. Boundary-aware test comparison: Compare HED, DQN, and PPO on per-sample boundary F1 over the 200-image test split. See runs/seed486_compare/test_compare_boundary_f1.png.
4. Binary quality comparison: Compare HED, DQN, and PPO on per-sample binary F1 and IoU. See runs/seed486_compare/test_compare_binary_f1.png and runs/seed486_compare/test_compare_iou.png.
5. Mean metric bars: Summarize average boundary F1, binary F1, and IoU across HED, DQN, and PPO for seed 486. See runs/seed486_compare/mean_boundary_f1_bar.png, runs/seed486_compare/mean_binary_f1_bar.png, and runs/seed486_compare/mean_iou_bar.png.

## Brief Interpretation

For seed 486, DQN is close to the HED baseline on binary F1 (0.4088 vs 0.4117) and IoU (0.2591 vs 0.2614), while PPO is lower on both metrics. On boundary F1, DQN (0.7515) slightly improves over PPO (0.7262) but remains below HED (0.7551).

## All-seed Aggregate

Across seeds 24, 42, 123, 486, and 999, DQN boundary F1 mean +/- std is 0.7508 +/- 0.0010, and PPO is 0.7449 +/- 0.0096.
