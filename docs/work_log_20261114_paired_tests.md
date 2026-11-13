# 2026-11-14 작업 기록: Table 4 점검 및 paired 통계 검정

리뷰어 코멘트 5 ("200장 test 이미지에서 HED-ODS vs. DQN, HED-ODS vs. PPO, DQN vs. PPO paired 검정, effect size와 95% CI 보고")에 대응하기 위한 작업입니다.

## 1. Table 4 점검

- Table 4(Reward Mean/Max/Std)는 **training 중 값이 아니라 test(200장) 평가 값**입니다.
  - 근거: `runs/all_seed_test_summary.csv` 및 각 run의 `test_summary.json`과 값이 일치합니다. 학습 중 값은 `csv_logs/eval_compare_metrics.csv`에 따로 있고 값이 다릅니다.
- 수정 필요 사항
  - 캡션의 "during training"을 test set 기준으로 변경 (예: "... on the test set for all seed cases").
  - seed 999 PPO Reward Mean: 22.40 → **22.41** (원본 22.4095).

## 2. paired 통계 검정

- 스크립트: [`scripts/paired_statistical_tests.py`](../scripts/paired_statistical_tests.py)
- 결과: [`runs/paired_stats/paired_tests.md`](../runs/paired_stats/paired_tests.md), [`runs/paired_stats/paired_tests.csv`](../runs/paired_stats/paired_tests.csv)
- 실행: `python scripts\paired_statistical_tests.py` (옵션: `--n-boot`, `--random-seed`)

### 설계

- 시드별 보고, 주 지표는 boundary F1과 IoU.
- 비교 쌍: HED-ODS vs. DQN, HED-ODS vs. PPO, DQN vs. PPO (차이 = 뒤 방법 − 앞 방법), 200장 test 이미지 기준.
- 보고 항목: 평균 차이와 bootstrap 95% CI(10,000회), Cohen's d_z, rank-biserial, Wilcoxon signed-rank p, paired t p, Holm 보정 p(seed·지표별 3개 비교).
- 입력 데이터(이미지별 지표)
  - HED-ODS: `runs/ods_binary_maps/test/rows.csv`
  - DQN/PPO: 각 run의 `test_policy_rows.csv` (seed 42 DQN은 `dqn_sqrt8/d_test_policy_rows.csv`, seed 486은 `*_seed486_sqrt8/test_eval/test_rows.csv`)
  - 스크립트가 각 run의 이미지별 평균이 `runs/all_seed_test_summary.csv`와 일치하는지 자동 검증합니다. 처음에 사용한 `internal_test_param_seed486_*`는 요약과 달라서 `test_eval`로 교체했습니다.

### 결과 요약

- HED-ODS 대비 DQN/PPO는 모든 seed에서 boundary F1, IoU가 약간 낮고(대부분 0.002~0.009, seed 486 PPO는 −0.029/−0.017) Holm 보정 후에도 유의합니다. 즉 RL이 baseline을 유의하게 개선했다고 주장할 수 없습니다.
- DQN과 PPO의 우열은 seed마다 갈립니다. 일관된 우위는 없습니다.

### 주의 사항

- PPO의 seed 42/123, seed 24/999 결과가 서로 동일합니다(예: 0.7521/0.2590, 0.7470/0.2597). 체크포인트 재사용 여부 확인이 필요합니다. 일부 비교의 d_z가 매우 큰 것은 이 때문일 수 있습니다.

## 3. "HED-ODS"의 정의

두 가지가 다릅니다.

| | 이 저장소의 ODS (paired 분석 사용) | 표준 BSDS-ODS |
|---|---|---|
| threshold 선택 | validation 100장에서 선택, 0.5 | test GT로 선택, 0.55 |
| 평가 | 자체 지표(boundary F1, IoU, F1) | BSDS500 benchmark (tolerance 0.0075 x 대각선) |
| 결과 | boundary F1 0.7551, IoU 0.2614 | ODS F 0.7540, OIS F 0.7710 |
| 위치 | `runs/ods_binary_maps/` | `runs/bsds_hed_ods_test/` |

- 논문에서는 "HED (validation-tuned global threshold)" 등으로 정의를 명시해 BSDS-ODS와 혼동되지 않게 합니다.
- BSDS-ODS는 test GT로 threshold를 고르는 참조값입니다.

## 4. 후속 과제 (미수행)

- DQN/PPO를 BSDS 평가기로 평가하는 것은 가능합니다. 이진맵이므로 단일 threshold(동작점) 한 점의 P/R/F만 정의되고 ODS/AP는 정의되지 않습니다. HED도 threshold 0.5로 이진화해 같은 평가기로 맞춰야 공정합니다. seed 486의 이진맵 존재 여부 확인이 필요합니다.
- PPO seed 간 동일 결과의 원인 확인.
- 논문 반영: Methods에 검정 방법 추가, Table 4 수정, paired 결과 표(Table 5 또는 부록) 추가, rebuttal에 반영 위치 기재.
