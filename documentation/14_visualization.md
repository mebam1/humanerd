# 학습 GT와 모델 rollout 시각화

학습 데이터가 어떤 접촉 사건을 포함하는지 확인하고, 모델이 정답과 언제부터 다른 움직임을 만드는지 영상과 시계열로 비교한다. **GT는 저장된 simulator 상태의 재생**, **모델 rollout은 checkpoint가 예측한 상태의 재귀적 전개**다. 이 명령은 학습을 시작하지 않는다.

기존 `render`는 이미 저장된 NPZ와 scene JSON을 읽어 영상으로 만드는 기능이었다. 대규모 학습 데이터의 NPY shard에서 trajectory를 고르고, checkpoint만으로 모델 rollout을 생성하는 진입점을 추가했다. `evaluate`의 `save_rollouts: false`와 독립적으로 사용할 수 있다.

## 구현한 구성 요소와 이유

| 구성 요소 | 구현 내용 | 필요한 이유 |
|---|---|---|
| Trajectory 선택 | `list-trajectories`, split·scenario·trajectory ID 필터 | 학습/평가 데이터를 구분하고 같은 사례를 반복 비교 |
| 공통 데이터 로더 | 기존 NPZ와 memory-mapped NPY shard에서 선택된 trajectory 읽기 | 13,000개 전체 데이터를 복사하거나 펼치지 않고 확인 |
| GT 재생 | `visualize-data`로 저장된 상태를 MuJoCo에서 렌더링 | 새 simulation의 수치 차이 없이 학습에 사용한 정답을 확인 |
| Checkpoint rollout | `visualize-rollout`에서 초기 상태와 기록된 action으로 자율 예측 | one-step 정확도만으로 드러나지 않는 누적 오차를 관찰 |
| 동기화 비교 영상 | 같은 카메라·시간의 GT와 예측을 나란히 렌더링 | 카메라 추적이나 재생 시점 차이를 모델 오차로 오해하지 않기 |
| 시계열·지표 | 상태 곡선, GT 대비 오차와 JSON 기록 | 육안 확인과 수치 비교를 함께 수행 |
| HTML 갤러리·재생 데이터 | 영상·그래프·선택 metadata와 trajectory/scene 파일을 출력 | localhost에서 편하게 비교하고 같은 결과를 다시 렌더링 |

영상의 물리 장면은 MuJoCo가 그린다. Warp로 생성한 데이터도 manifest와 저장된 상태를 읽어 동일한 렌더링 경로로 볼 수 있다. 여기서 `--device cuda:0`은 **모델 추론 장치**이고, `MUJOCO_GL=osmesa`는 화면 없이 동작하는 **CPU 영상 렌더러** 설정이다. CUDA 추론과 CPU 렌더링을 함께 사용할 수 있다.

## 1. 저장된 학습 데이터에서 GT 선택

저장소 루트에서 실행한다. 새 주 데이터 `data/cube-toss`는 랜덤 초기 위치·회전·선속도·각속도의 train 투척 10,000개, validation/test 투척 각 1,000개를 포함한다. 모든 Cube action은 0이다. 별도 OOD 1,000개는 바닥–벽 동시 접촉 진단 초기 조건이다.

```bash
docker compose run --rm research python -m cube list-trajectories --dataset data/cube-toss --split train --scenario toss --limit 10
docker compose run --rm -e MUJOCO_GL=osmesa research python -m cube visualize-data --dataset data/cube-toss --split train --scenario toss --limit 6 --output docs/visualizations/gt-toss
docker compose up -d docs
```

[새 투척 GT 갤러리 열기](visualizations/gt-toss/index.html) · [초기 조건과 action 정책](15_cube_toss.md)

`visualize-data`는 기본적으로 train split을 선택하며 trajectory의 전체 저장 구간을 재생한다. 기본 데이터는 100 transitions·101 states로 2초 길이다. `--horizon 30`처럼 앞부분만 선택할 수도 있다. `--limit 4`는 영상으로 만들 trajectory 수이며 학습 데이터 자체를 변경하지 않는다. 목록에서 확인한 ID를 `--trajectory-id`에 전달하면 해당 사례를 선택한다. 명시한 split·scenario 필터가 있다면 ID도 그 조건과 일치해야 한다. 다른 split의 ID를 선택하려면 해당 split을 명시하거나 split 필터를 생략한다.

| 관찰 목적 | Split | Scenario |
|---|---|---|
| 랜덤 위치·회전의 비행, 회전, 착지와 반발 | 새 `data/cube-toss` train 또는 test | `toss` |
| 학습에 없는 동시 접촉 진단 | 새 `data/cube-toss` ood | `floor_wall` |
| 과거 데이터의 충돌 전후 높이와 반발 | 보존한 기존 시나리오 갤러리 | `impact`, `rebound` |
| 과거 데이터의 비접촉 운동 | 보존한 기존 시나리오 갤러리 | `free_fall` |
| 과거 데이터의 접촉·마찰 | 보존한 기존 시나리오 갤러리 | `rest`, `settle`, `slide` |
| 과거 데이터의 회전과 벽 접촉 | 보존한 기존 시나리오 갤러리 | `edge`, `wall_only` |

`--scenario`를 생략하면 split 내 모든 시나리오가 선택 대상이 된다. 한두 영상은 데이터 분포 전체를 대표하지 않으므로 여러 시나리오를 따로 확인한다.

기존 `impact` ID 1500–1503은 모두 단위 회전, 수평 위치 0, 초기 각속도 0인 거의 수직 낙하 사례다. 이는 새 투척의 랜덤 초기 회전 분포가 아니다. 다른 기존 움직임은 [9개 시나리오 GT 비교](visualizations/gt-scenarios/index.html)에서 볼 수 있다. 기존 데이터에는 일부 시나리오의 nonzero action도 있으므로 과거 영상과 새 무구동 투척을 구별한다.

## 2. Checkpoint 자율 rollout의 의미

기존 [모델 rollout 비교 갤러리](visualizations/model-impact/index.html)는 이전 `data/cube-nerd`에서 실행한 **32 trajectories·3 epochs 검증용 계층형 모델** 기록이다. 새 투척 데이터로 학습한 모델이 아니다. 기존 원본 데이터는 새 데이터 검증 후 제거했으며, 만들어 둔 갤러리·checkpoint는 과거 기록으로 남긴다. 새 모델의 명령은 다음 절을 사용한다.

새 데이터로 실제 학습한 **16 trajectories·5 epochs 검증 모델**도 있다. 아래 명령은 그 checkpoint로 시각화하며 학습을 다시 시작하지 않는다. [학습 검증 결과](15_cube_toss.md)처럼 데이터·메인 모델의 동작 확인용이고 정확한 장기 예측 모델은 아니다.

```bash
docker compose run --rm -e MUJOCO_GL=osmesa research python -m cube visualize-rollout --dataset data/cube-toss --checkpoint outputs/toss-validation/hierarchical/best.pt --split test --scenario toss --limit 2 --horizon 100 --device cpu --output docs/visualizations/model-toss-validation
```

[새 투척 검증 모델의 GT/rollout 비교](visualizations/model-toss-validation/index.html)

`visualize-rollout`은 기본 test split을 사용한다. 모델과 GT는 같은 초기 상태에서 출발하고 같은 기록 action을 사용한다. 이후 모델 입력에는 자기 예측 상태가 들어가며, 미래 GT 상태로 교체하지 않는다. history 모델도 예측 이력을 사용한다. 따라서 이 영상은 teacher forcing된 one-step 예측 영상과 다르다. action을 새로 계획하는 MPC 실험도 아니며, 기록된 action에 대한 동역학 예측을 비교한다.

`--horizon 100`은 최대 100번의 상태 전개를 요청한다. trajectory에 저장된 action 수를 넘어서 생성하지 않는다. 장면과 물체 종류가 호환되는 dataset/checkpoint를 선택한다. 다른 물체로의 전이는 해당 전이 프로토콜로 학습한 checkpoint를 사용한다.

## 3. 5-epoch 학습 모델과 다른 구조 비교

현재 suite는 모델·seed별 **5 epochs, gradient norm clipping 1.0**이다. 학습이 완료되어 `best.pt`가 만들어진 후 아래 경로를 사용할 수 있다. `best.pt`는 validation으로 선택한 모델이며 마지막 epoch와 같다고 보장하지 않는다.

```bash
docker compose --profile cuda run --rm -e MUJOCO_GL=osmesa cuda python -m cube visualize-rollout --dataset data/cube-toss --checkpoint outputs/cube-toss-suite-5ep/hierarchical_budget10000_seed0/best.pt --split test --scenario toss --limit 6 --horizon 100 --device cuda:0 --output docs/visualizations/hierarchical-toss-5ep
docker compose --profile cuda run --rm -e MUJOCO_GL=osmesa cuda python -m cube visualize-rollout --dataset data/cube-toss --checkpoint outputs/cube-toss-suite-5ep/state_mlp_budget10000_seed0/best.pt --split test --scenario toss --limit 6 --horizon 100 --device cuda:0 --output docs/visualizations/state-mlp-toss-5ep
```

다른 구조도 checkpoint 경로의 모델 이름을 `deepsets`, `flat`, `independent`, `nerd_history1`, `nerd_history10`으로 바꾸고 출력 폴더를 분리한다. 비교 모델마다 **같은 dataset·trajectory ID·horizon**을 선택한다. 갤러리의 선택 metadata에서 ID를 확인할 수 있다. seed 1, 2도 동일한 조건으로 확인하며, 보기 좋은 사례만으로 모델 우열을 판단하지 않는다.

단독 `train`으로 생성했다면 checkpoint는 `outputs/cube-toss-gpu-5ep/hierarchical/best.pt`다. 모델 경로만 바꾸면 같은 명령을 사용할 수 있다. CUDA 없이 추론하려면 `research` 서비스와 `--device cpu`를 사용한다. CPU 추론은 모델 크기와 rollout 길이에 따라 시간이 걸릴 수 있다.

## 4. 학습에 없는 동시 접촉 OOD 확인

```bash
docker compose --profile cuda run --rm -e MUJOCO_GL=osmesa cuda python -m cube visualize-rollout --dataset data/cube-toss --checkpoint outputs/cube-toss-suite-5ep/hierarchical_budget10000_seed0/best.pt --split ood --scenario floor_wall --limit 2 --horizon 100 --device cuda:0 --output docs/visualizations/model-toss-ood
```

명령을 실행한 뒤 `http://localhost:8000/visualizations/model-toss-ood/index.html`에서 확인한다. 이 OOD는 기존 `floor_wall` 접촉 진단 분포이며 초기 위치·회전·속도를 모두 무작위화한 투척 OOD가 아니다.

바닥–벽 접촉 시점, 회전 방향, 튀는 높이, 관통이나 장기 드리프트를 확인한다. 영상의 사례별 오차는 전체 test/OOD 평균과 다르다. 정량 결론은 [GPU 학습 안내](12_gpu_training.md)의 `evaluate` 명령으로 같은 평가 subset 또는 전체 1,000 trajectories를 측정한 결과와 함께 판단한다.

## 영상 크기·재생·문서 서버

기본 선택 명령 뒤에 `--width 640 --height 480 --stride 2`를 추가하면 출력 해상도와 저장 프레임 수를 조절할 수 있다. 640 × 480은 패널 하나의 크기로, 비교 영상은 가로 1,280픽셀이다. `--stride`는 모델 계산의 timestep이나 rollout 길이를 바꾸지 않는다. trajectory의 시간 간격에 맞춰 영상 FPS를 정하므로 프레임을 건너뛰더라도 재생 속도는 유지된다.

| 출력 파일 | 내용 |
|---|---|
| `index.html` | 선택한 trajectory의 영상과 그래프를 모은 갤러리 |
| `manifest.json` | dataset·checkpoint·선택 조건과 생성 결과 |
| `trajectory_XXXXX/gt.mp4` | GT 재생 영상 |
| `trajectory_XXXXX/comparison.mp4` | 모델 rollout 명령의 GT/예측 비교 영상 |
| 영상과 이름이 같은 `.png` | 영상 포스터 |
| `trajectory_XXXXX/timeseries.png` | 상태 시계열 및 rollout 오차 그래프 |
| `trajectory_XXXXX/metrics.json` | 사례별 수치와 진단 정보 |
| `trajectory_XXXXX/trajectory.npz` | GT는 `states/actions`, rollout은 `truth/predicted/actions` 배열 |
| `trajectory_XXXXX/scene.json` | 원본 trajectory의 물리 장면 설정 |

GT와 rollout 명령은 각각 필요한 영상만 만든다. `XXXXX`는 0으로 채운 trajectory ID다. 예측이 발산했다면 갤러리의 상태와 유효 구간을 함께 확인한다. 영상에 보이는 짧은 정상 구간만으로 끝까지 안정적이었다고 해석하지 않는다.

`ROLLOUT COMPLETE`는 요청 구간의 수치 계산이 끝났다는 뜻이며 물리적 정확성을 보장하지 않는다. 예측이 바닥 아래로 내려가거나 GT 기준 카메라 밖으로 벗어나도 보정하지 않는다. 상태 그래프와 오차로 해당 움직임을 확인할 수 있다.

## 실제 생성한 예시와 관찰

새 [투척 GT 갤러리](visualizations/gt-toss/index.html)는 `data/cube-toss`의 train/toss를 보여 준다. 기존 [impact GT 갤러리](visualizations/gt-impact/index.html)는 train/impact ID **1500–1503**, 기존 모델 비교 갤러리는 test/impact ID **2875–2876**의 기록이다. 기존 영상은 100 transitions, 물리 시간 2초이며 `--stride 2`로 저장했다. GT는 640 × 480, 비교 영상은 1280 × 480이다.

사용한 모델은 위의 32 trajectories·3 epochs 검증용 checkpoint다. ID 2875에서 모델의 높이가 시간이 지나며 바닥 아래로 내려가는 오류가 보인다. 100-step의 root position RMSE는 **2.382 m**로, 긴 구간의 접촉 예측이 부정확하다. RMSE는 세 좌표 성분에 대한 평균 제곱 오차의 제곱근이며, 그래프에 표시한 시각별 Euclidean 거리와는 정의가 다르다. 이 결과는 시각화가 누적 오차를 드러내는 예시이며, 10,000 trajectories·5 epochs 학습 결과나 모델 구조 간 비교 결론이 아니다.

출력 폴더를 `docs/visualizations/` 아래로 지정하면 기존 문서 서버로 바로 볼 수 있다. 갤러리는 실행 시 자체 `index.html`을 만들고, 문서 메뉴를 갱신하는 명령은 다음과 같다.

```bash
docker compose run --rm research python -m cube docs
docker compose up -d docs
docker compose logs docs
```

기본 주소는 [http://localhost:8000](http://localhost:8000)이다. `.env`에서 `DOCS_PORT`를 변경했다면 해당 포트를 사용한다. 저장소 루트가 `/workspace`에 bind mount되므로 출력 파일은 컨테이너 종료 뒤에도 호스트에 남는다. 시각화는 학습이나 W&B run을 시작하지 않으며 W&B 인증도 필요하지 않다.

## 기존 NPZ 영상 재생

Ant 제어 실험이나 평가에서 이미 저장한 NPZ·scene JSON에는 기존 명령을 그대로 사용할 수 있다.

```bash
docker compose run --rm -e MUJOCO_GL=osmesa research python -m cube render --trajectory outputs/ant_walk/rollout.npz --scene outputs/ant_walk/scene.json --output docs/assets/ant-walking.mp4 --stride 2
```

이 명령은 NPZ의 `states` 배열을 렌더링한다. 다른 키로 저장된 예측 상태를 보려면 `--key`로 배열 이름을 지정한다. 저장된 모델 예측을 다시 그릴 때에도 MuJoCo는 장면 표시만 담당하며, 접촉 동역학을 다시 적분해 예측을 보정하지 않는다.
