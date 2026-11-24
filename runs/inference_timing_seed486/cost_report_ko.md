# 이미지당 후처리 시간 측정 결과

- 측정일: 2026-11-24
- 대상: `inputs`의 test split 200장. HED map 크기는 321 × 481 픽셀 134장, 481 × 321 픽셀 66장이다.
- 각 방법을 5장으로 워밍업한 뒤, 같은 200장을 각각 5회 측정했다. 워밍업에 사용한 이미지도 본 측정에 포함했다.
- DQN과 PPO는 재학습하지 않고 seed 486의 최종 체크포인트를 사용했다.

## 결과

단위는 이미지당 ms이다. 각 이미지의 5회 측정값을 먼저 평균한 뒤, 그 200개 평균의 평균과 모집단 표준편차(`ddof=0`)를 계산했다. 따라서 아래 표준편차는 seed 간 또는 회차 간 표준편차가 아니라 **이미지별 5회 평균의 표준편차**이다.

| 방법 | 평균 ± 표준편차 (ms/이미지) |
|---|---:|
| HED-ODS | 0.093 ± 0.008 |
| HED-Otsu | 0.966 ± 0.079 |
| DQN | 220.355 ± 2.907 |
| PPO | 223.948 ± 2.555 |

### 회차별 평균

각 숫자는 해당 회차의 200장 평균 시간(ms/이미지)이다. 한 프로세스에서 연속 측정한 회차로, 별도 프로세스 재시작이나 독립적인 장비 실험은 아니다.

| 방법 | 1회 | 2회 | 3회 | 4회 | 5회 | 회차 평균의 표준편차 |
|---|---:|---:|---:|---:|---:|---:|
| HED-ODS | 0.085 | 0.093 | 0.097 | 0.095 | 0.092 | 0.004 |
| HED-Otsu | 0.910 | 0.960 | 0.986 | 0.999 | 0.974 | 0.031 |
| DQN | 214.900 | 217.873 | 219.061 | 226.631 | 223.310 | 4.140 |
| PPO | 217.969 | 221.693 | 221.903 | 231.400 | 226.776 | 4.659 |

회차 평균의 표준편차 역시 모집단 표준편차(`ddof=0`)이다. 회차별 차이는 존재하므로 위 숫자는 이 실행 환경에서의 측정값으로 해석해야 한다.

## 측정 범위 및 환경

- 모든 방법은 **메모리에 미리 적재된 HED score map에서 이진 edge map을 생성하는 후처리 시간**을 측정했다. HED 신경망 실행 시간은 포함하지 않는다. 따라서 표는 원본 이미지부터 최종 결과까지의 전체 추론 시간이 아니다.
- 입력 이미지와 HED map 읽기, 출력 저장, GT 평가, 모델 로딩 및 ODS threshold의 사전 탐색은 측정에서 제외했다.
- HED-ODS는 validation에서 선택한 고정 threshold 0.5를 사용했다. 이는 본 프로젝트의 practical ODS이며 엄밀한 BSDS benchmark ODS와 구분해야 한다.
- HED-Otsu는 이미지별 threshold 계산을 포함한다. DQN과 PPO는 GT-free 환경의 초기화, 관측 특징 계산, 결정적 정책 예측, 30 step의 반복 후처리와 파라미터 기록을 포함한다.
- CPU: Intel Core i9-14900K; GPU 미사용. PyTorch intra-op 및 inter-op 스레드 수는 각각 1이다. 측정 중 다른 프로세스의 부하는 통제하지 않았다.
- Windows 11, Python 3.14.1, PyTorch 2.11.0+cpu, NumPy 2.4.4, Pillow 12.2.0, Stable-Baselines3 2.8.0.
- `time.perf_counter_ns()`로 이미지별 경과 시간을 측정했다. 방법의 실행 시작 순서는 이미지와 회차에 따라 순환시켰다.
- 체크포인트:
  - DQN: `runs\dqn_seed486_sqrt8\dqn_final.zip`; SHA-256 `78c0c62b6fff9491b32d6404c087534cf0a389a3d4c2110ffd593f58e7d3ecdf`
  - PPO: `runs\ppo_seed486_sqrt8\ppo_final.zip`; SHA-256 `0ec7e65e5b234a49e5b78598de675bc5fde4173bcaee51c8fe650c870b6be36`

## 결과 검증 및 자료

- 각 방법의 출력은 같은 이미지에 대해 5회 모두 동일했다.
- ODS와 Otsu 출력 200장은 기존에 저장된 이진 맵과 픽셀 단위로 일치했다.
- DQN과 PPO 출력의 F1과 IoU는 seed 486 최종 모델의 기존 test 결과와 이미지별로 일치했으며, 모두 30 step을 수행했다. 이는 기존 RL 이진 맵과의 픽셀 단위 비교를 의미하지는 않는다.
- 원시 측정값: [inference_timings.csv](./inference_timings.csv), 회차별 200장씩 총 1,000행.
- 이미지별 5회 평균: [per_image_means.csv](./per_image_means.csv), 총 200행.
- 전체 정밀도 집계 및 환경: [inference_timing_summary.json](./inference_timing_summary.json).
- 재현 코드: [benchmark_inference.py](./benchmark_inference.py). 저장소 루트에서 아래 명령을 실행하면 같은 폴더의 CSV와 JSON을 덮어쓴다. 결과 보고서는 자동 갱신되지 않으므로 재측정 후 별도로 갱신해야 한다.

```powershell
& .\.venv\Scripts\python.exe .\runs\inference_timing_seed486\benchmark_inference.py
```

## 논문 표기 예시

> HED score map에 대한 이미지당 후처리 시간은 test 이미지 200장 각각을 5회 측정한 후 이미지별 평균을 구하여 산출하였다. CPU 환경에서 HED-ODS, HED-Otsu, DQN, PPO의 평균 ± 이미지 간 표준편차는 각각 0.093 ± 0.008, 0.966 ± 0.079, 220.355 ± 2.907, 223.948 ± 2.555 ms였다. HED 네트워크 추론과 파일 입출력은 제외하였다.

## HED 예측 포함 지연 시간: 제공받은 벤치마크

다음은 사용자가 제공한 별도 벤치마크 결과이며, 이 저장소에서 독립적으로 재측정한 값은 아니다.

| 측정 항목 | 평균 | 중앙값 | p95 |
|---|---:|---:|---:|
| HED 추론 함수 `infer` | 370.4 ms | 369.8 ms | 388.3 ms |
| 이미지 파일 읽기 + HED 추론 | 372.4 ms | 371.8 ms | 390.1 ms |

- 처리량: 약 2.69장/초.
- 모델 초기 로딩: 89.8 ms, 1회 측정.
- `inputs_old/test`의 이미지 200장으로 측정했으며, 첫 추론은 워밍업으로 제외했다.
- `infer`에는 이미지 패딩·입력 변환, HED 순전파, 출력 크기 보정과 후처리가 포함된다. 따라서 370.4 ms는 순전파 연산만의 시간이 아니다.
- 입력 파일 읽기 시간은 별도로 더해 372.4 ms를 산출했다. 이 값은 **이미지 읽기부터 HED 출력까지**의 지연 시간이며, 위 표의 ODS·Otsu·DQN·PPO 후처리 시간은 포함하지 않는다.
- 제공된 측정 환경: 프로젝트 가상환경의 OpenCV 4.8.1, OpenCV 스레드 32개, CUDA backend 미지정. 패딩은 1.5배 확대, 최소 여백 30 px로 설정되었다.
- 제공된 코드 참조는 `hed_inference.py`의 23–24행과 71행이다. 해당 파일과 `inputs_old/test`는 현재 저장소에서 확인되지 않았으므로 코드, 데이터 및 기존 HED 맵과의 동일성은 독립적으로 검증하지 못했다.
- 위 후처리 벤치마크는 별도 환경·절차로 실행되었고 입력 읽기를 제외하므로, 두 결과를 단순 합산한 값을 네 방법의 실측 end-to-end 지연 시간이나 표준편차로 보고하지 않는다. HED 출력부터 각 방법의 최종 출력까지 함께 측정해야 한다.

## 시드별 학습 시간: 기존 로그 기반 근사치

기존 TensorBoard 이벤트 파일의 **생성 이벤트 시각부터 마지막 `time/fps` 기록 시각까지**의 wall-clock 경과 시간을 계산했다. 이는 `model.learn()` 전체 시간의 직접 측정값이 아니며, 중간 validation과 로깅을 포함한 경과 시간이다. 학습 프로세스 초기화 일부와 마지막 로그 이후 작업은 빠진다.

| seed | DQN 경과 시간 (분) | DQN 마지막 기록 step | PPO 경과 시간 (분) | PPO 마지막 기록 step |
|---:|---:|---:|---:|---:|
| 24 | 148.66 | 199,915 | 149.39 | 200,192 |
| 42 | 164.68 | 198,899 | 121.19 | 200,192 |
| 123 | 160.37 | 199,889 | 119.83 | 200,192 |
| 486 | 144.14 | 199,983 | 140.55 | 200,192 |
| 999 | 146.74 | 199,982 | 141.74 | 200,192 |

- PPO 5개 시드의 **로그 구간 경과 시간**: **134.54 ± 11.86분** (시드 간 모집단 표준편차, `ddof=0`). 목표가 200,000 step인 PPO는 rollout 단위 때문에 각 최종 체크포인트에서 200,192 step으로 기록되어 있다. 마지막 로그 이후 작업은 이 시간에 포함되지 않는다.
- DQN 5개 로그 구간의 단순 집계는 **152.92 ± 8.09분**이지만, 아래 seed 42 문제와 각 로그의 마지막 step 차이 때문에 이를 **200,000 step 학습 완료 시간**이라고 표기할 수 없다. seed 42를 제외한 나머지 4개 로그 구간의 집계는 149.98 ± 6.21분이다.
- DQN seed 42는 최종 `time/fps` 기록이 198,899 step이고, `training_metrics.csv`는 199,000 step, `eval_compare_metrics.csv`는 198,000 step까지 기록되어 있다. 저장된 `dqn_final.zip` 및 `best_model.zip`의 `num_timesteps`도 각각 184,000이다. 따라서 이 자료로는 seed 42의 200,000 step 완료 또는 완료 시간을 확인할 수 없다.
- 다른 DQN 시드 4개의 최종 체크포인트에는 200,000 step이 기록되어 있지만, TensorBoard 마지막 기록은 그보다 앞서므로 표의 경과 시간에 마지막 기록 이후의 학습 및 validation은 포함되지 않는다.
- seed 24, 486, 999의 DQN과 PPO 이벤트 기록 구간은 각각 서로 겹친다. 프로세스는 같은 호스트명으로 기록되어 있어 동시 실행에 따른 자원 경합 가능성이 있으며, 단독 실행 환경의 속도로 해석할 수 없다. 당시 CPU/GPU와 스레드 설정은 이 이벤트 기록만으로 확인하지 못했다.

따라서 정확한 200,000 step 학습 소요시간의 5개 시드 평균 ± 표준편차가 논문에 필요하다면, 동일한 장비와 조건에서 5개 시드를 재학습하며 `model.learn()` 시작과 종료 시각을 직접 기록해야 한다. 위 근사치는 재학습 없이 기존 기록에서 복원한 참고값이다.

### 로그 출처

- seed 42: [DQN TensorBoard](../dqn_sqrt8/tensorboard/DQN_1/), [PPO TensorBoard](../ppo_sqrt8/tensorboard/PPO_1/).
- seed 24: [DQN TensorBoard](../dqn_seed24_sqrt8/tensorboard/DQN_1/), [PPO TensorBoard](../ppo_seed24_sqrt8/tensorboard/PPO_1/).
- seed 123: [DQN TensorBoard](../dqn_seed123_sqrt8_fresh/tensorboard/DQN_1/), [PPO TensorBoard](../ppo_seed123_sqrt8_fresh/tensorboard/PPO_1/).
- seed 486: [DQN TensorBoard](../dqn_seed486_sqrt8/tensorboard/DQN_1/), [PPO TensorBoard](../ppo_seed486_sqrt8/tensorboard/PPO_1/).
- seed 999: [DQN TensorBoard](../dqn_seed999_sqrt8/tensorboard/DQN_1/), [PPO TensorBoard](../ppo_seed999_sqrt8/tensorboard/PPO_1/).
