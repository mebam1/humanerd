# Cube Contact Dynamics

연구 제안의 **Point → Face → Body / Point → Patch → Link → Articulation** 동역학 모델을 Cube 접촉 실험부터 사족 보행 Ant까지 실행하는 연구 코드입니다. 시뮬레이터는 **MuJoCo 3.16.0**과 **MuJoCo Warp 3.16.0 / Warp 1.18.0**입니다.

저장소 루트가 컨테이너의 `/workspace`에 bind mount됩니다. 코드 변경과 `data/`, `outputs/`의 실험 결과는 호스트에 그대로 남습니다. 호스트 Python 설치 없이 아래 Docker Compose 명령으로 실행합니다.

## 시작하기

Docker Desktop의 Linux containers가 필요합니다. GPU 서비스는 NVIDIA GPU와 Docker GPU 지원이 필요합니다. 기본 CPU 서비스는 GPU 없이 실행할 수 있습니다.

```bash
docker compose build research
docker compose run --rm research python -m cube doctor
docker compose run --rm research python -m cube --help
docker compose up -d docs
```

**[HTML 문서 대시보드](http://localhost:8000)** · **[직접 실행한 실험 결과](http://localhost:8000/10_results.html)**

문서에는 구현한 논리적 구성 요소, 설계 이유, 좌표계, 실험 프로토콜, 측정 그래프와 Ant 보행 영상이 있습니다. HTML은 `docs/`, 편집 가능한 문서 원본은 `documentation/`에 있습니다.

```bash
docker compose run --rm research python -m cube docs
docker compose logs docs
docker compose down
```

문서 서비스는 `127.0.0.1:8000`에만 노출됩니다. `.env`의 `DOCS_PORT=8080`으로 포트를 바꿀 수 있습니다.

## NeRD 규모의 Cube GPU 학습과 구조 비교

**[단계별 GPU 실험 안내](http://localhost:8000/12_gpu_training.html)**에 구성 요소·설계 이유·수렴 판단·학습 재개·최종 평가 절차를 정리했습니다.

주 학습 데이터는 **랜덤 위치·회전·선속도·각속도의 Cube 투척**으로 바꿨습니다. 투척 후 action은 모든 시점에서 0이며, 중력과 접촉으로 움직입니다. **[투척 데이터 구성과 기존 데이터 감사](http://localhost:8000/15_cube_toss.html)** · **[새 투척 GT 갤러리](http://localhost:8000/visualizations/gt-toss/index.html)**에서 확인합니다. 새 데이터 검증 후 기존 `data/cube-nerd` 원본을 제거했습니다. 감사 결과·갤러리·과거 checkpoint는 기록으로 남겼습니다. 짧은 7개 모델 CUDA 비교와 재개 검증인 **[GPU 실행 결과](http://localhost:8000/13_gpu_results.html)**는 이전 데이터의 3-epoch 검증 기록입니다.

NeRD **Appendix C.4의 Cube 전용 synthetic 데이터는 랜덤 초기 상태의 투척 10,000 trajectories × 길이 100**입니다. 새 `configs/cube_toss_nerd_scale.json`은 학습용 투척 10,000개, validation/test 투척 각 1,000개, 별도 OOD 바닥–벽 동시 접촉 1,000개를 생성합니다. OOD는 기존 접촉 진단 초기 조건이며 랜덤 투척 분포가 아닙니다. 이 저장소의 길이는 100 transitions와 101 states입니다. 초기 분포의 구체적인 범위는 자체 설정이며 원본 NeRD의 모델·solver·실측 데이터 재현은 아닙니다. [원논문](https://arxiv.org/html/2508.15755v1#A3.SS4)

새 13,000개 생성과 전수 검증을 완료했고 action은 모두 0입니다. 메인 계층형 구조 그대로 16개 trajectory를 GPU에서 5 epochs 학습해 로더·정규화·학습·clipping 경로도 검증했습니다. 이 작은 검증 모델은 아직 정확도가 낮으며 10,000개 전체 학습 결과가 아닙니다. [새 데이터 학습 검증 결과](http://localhost:8000/15_cube_toss.html)와 [검증 모델 rollout](http://localhost:8000/visualizations/model-toss-validation/index.html)에서 한계까지 확인할 수 있습니다.

1. `.env.example`을 참고해 저장소 루트의 로컬 `.env`에 `WANDB_API_KEY`, `WANDB_ENTITY`를 설정합니다. API 키는 명령이나 JSON config에 넣지 않습니다.
2. 이미지를 빌드하고 `doctor`의 `torch_cuda: true`를 확인합니다.
3. 데이터를 생성한 뒤 계층형 모델을 학습하거나 전체 비교를 실행합니다.

```bash
docker compose --profile cuda build research cuda
docker compose --profile cuda run --rm cuda python -m cube doctor
docker compose run --rm research python -m cube data-plan --config configs/cube_toss_nerd_scale.json
docker compose run --rm research python -m cube generate --config configs/cube_toss_nerd_scale.json --output data/cube-toss --resume
docker compose run --rm research python scripts/audit_cube_toss.py --dataset data/cube-toss
docker compose run --rm research python scripts/verify_toss_model.py --dataset data/cube-toss --config configs/train_gpu.json --trajectories 40 --output docs/results/cube_toss_model_contract.json
docker compose --profile cuda run --rm cuda python -m cube train --dataset data/cube-toss --config configs/train_gpu.json --model hierarchical --seed 0 --output outputs/cube-toss-gpu-5ep/hierarchical
```

생성은 MuJoCo CPU worker 8개를 사용하고, 학습은 CUDA와 bf16을 사용합니다. 데이터는 NPY shard로 저장하며 필요한 부분만 읽습니다. 동일한 생성 버전·설정으로 만든 데이터의 `--resume`는 파일을 검증하고 재사용합니다. 중단된 생성은 완료된 shard부터 이어집니다. 기존 `data/cube-nerd`에 새 설정으로 `--resume`하지 마세요. 초기 상태·action 정책이 바뀌어 생성 fingerprint가 다르며 새 데이터와 새 학습 출력 폴더를 사용합니다.

기본 학습은 **모델·seed별 5 epochs, gradient L2 norm clipping 1.0, effective batch 512, LR 1e-3 → 1e-4**입니다. 이는 수렴을 보장하는 수치가 아니며 validation 추세를 확인해야 합니다. GPU 비교는 짧은 smoke보다 훨씬 오래 걸립니다. 먼저 아래 용량 측정으로 본인 GPU의 처리량과 메모리를 확인할 수 있습니다.

```bash
docker compose --profile cuda run --rm cuda python scripts/benchmark_gpu.py --output outputs/gpu_capacity.json
docker compose --profile cuda run --rm cuda python -m cube suite --config configs/suite_gpu.json --output outputs/cube-toss-suite-5ep --resume
```

Suite는 **7개 설정 × 3개 seed = 21회**를 비교합니다: `hierarchical`, `state_mlp`, `deepsets`, `flat`, `independent`, `nerd_history1`, `nerd_history10`. 같은 학습 데이터·epoch·검증 subset을 사용하고 test/OOD 평가와 seed별 요약을 `suite.json`에 저장합니다. `nerd_history*`는 내부 비교군입니다. 전체 비교에서 계층형부터 보고 싶으면 같은 suite 명령에 `--models hierarchical`을 추가한 다음, filter 없이 `--resume`로 나머지를 진행합니다.

W&B online에 학습/검증 loss, LR, 처리량, GPU 메모리, test/OOD의 one-step·rollout·접촉 지표를 기록합니다. 링크와 run ID는 각 실행의 `wandb_run.json`에 남습니다. online 인증 실패는 오류로 처리합니다. 인증 없이 로컬 기록을 시험하려면 별도 출력 폴더와 `--wandb-mode offline`을 사용합니다. offline 로그는 온라인 전송 완료를 의미하지 않습니다.

전체 실행 전에 새 투척 32 trajectories·5 epochs의 짧은 CUDA 경로를 확인하려면 다음 별도 설정을 사용합니다. 같은 모델 크기를 사용하지만 본 비교의 성능 결론을 내리는 설정은 아닙니다. 이전 `outputs/gpu-check`의 3-epoch 기록과 구별해 새 출력 폴더를 사용합니다.

```bash
docker compose --profile cuda run --rm cuda python -m cube suite --config configs/suite_gpu_check.json --output outputs/gpu-check-toss --resume
```

학습 중단 후에는 동일 config·model·seed·output으로 재개합니다.

```bash
docker compose --profile cuda run --rm cuda python -m cube train --dataset data/cube-toss --config configs/train_gpu.json --model hierarchical --seed 0 --output outputs/cube-toss-gpu-5ep/hierarchical --resume outputs/cube-toss-gpu-5ep/hierarchical/last.pt
docker compose --profile cuda run --rm cuda python -m cube evaluate --checkpoint outputs/cube-toss-gpu-5ep/hierarchical/best.pt --dataset data/cube-toss --device cuda:0 --batch-size 8 --max-trajectories 1000 --rollout-horizon 100 --no-save-rollouts --output outputs/cube-toss-gpu-5ep/hierarchical/test-full
docker compose --profile cuda run --rm cuda python -m cube evaluate --checkpoint outputs/cube-toss-gpu-5ep/hierarchical/best.pt --dataset data/cube-toss --split ood --device cuda:0 --batch-size 8 --max-trajectories 1000 --rollout-horizon 100 --no-save-rollouts --output outputs/cube-toss-gpu-5ep/hierarchical/ood-full
docker compose run --rm research python -m cube coupling --checkpoint outputs/cube-toss-gpu-5ep/hierarchical/best.pt --output outputs/cube-toss-gpu-5ep/hierarchical/coupling
```

`best.pt`는 validation으로 선택한 평가용 모델, `last.pt`는 optimizer·scheduler·RNG·배치 진행 위치까지 복원하는 재개용입니다. 최적화 설정을 바꾸면 새 출력 폴더를 사용합니다. `epochs` 변경은 동일 학습 재개가 아닙니다. tqdm은 데이터 생성·정규화·학습·검증·평가·suite에서 진행률을 표시합니다. 로그만 필요한 환경에서는 Compose `run` 뒤에 `-e TQDM_DISABLE=1`을 붙입니다.

이전 데이터·50-epoch 실행은 새 투척 데이터의 5-epoch 실행으로 재개하지 않습니다. 위 명령은 새 `outputs/cube-toss-suite-5ep`를 사용합니다. Clipping은 AMP unscale·gradient accumulation 평균화 후 optimizer step 전에 global L2 norm 1.0으로 적용됩니다. W&B의 `train/gradient_norm`은 **clipping 전 값**입니다.

## 작은 실험 전체 재현

```bash
docker compose run --rm research python -m cube quick
```

이 명령은 상태 표현 검증, Cube 12회 학습 비교, A/B 접촉 개입, 두 링크 전이 8회, Ant 학습·평가·보행 제어, Cube MPC를 실행하고 결과 HTML을 갱신합니다. smoke 설정은 실행 검증용이며 충분히 학습된 모델이 아닙니다. 기존 결과가 있으면 새 경로 `--output outputs/quick_v2`로 실행합니다. 학습 결과를 묵시적으로 덮어쓰지 않습니다.

실제 CUDA Warp 검증까지 포함하려면:

```bash
docker compose --profile gpu run --rm warp python -m cube doctor
docker compose --profile gpu run --rm warp python -m cube quick --with-warp
```

저장된 결과에서 그래프·문서만 다시 만들기:

```bash
docker compose run --rm research python -m cube report --output outputs/quick
docker compose run --rm research python -m cube docs
```

## Cube 데이터·모델·평가

```bash
docker compose run --rm research python -m cube validate --output outputs/validation
docker compose run --rm research python -m cube generate --config configs/cube_smoke.json --output data/cube
docker compose run --rm research python -m cube train --dataset data/cube --config configs/train_smoke.json --model hierarchical --output outputs/cube
docker compose run --rm research python -m cube evaluate --checkpoint outputs/cube/best.pt --dataset data/cube --output outputs/cube/test
docker compose run --rm research python -m cube evaluate --checkpoint outputs/cube/best.pt --dataset data/cube --split ood --output outputs/cube/ood
docker compose run --rm research python -m cube coupling --checkpoint outputs/cube/best.pt --output outputs/cube/coupling
docker compose run --rm research python -m cube suite --config configs/suite_smoke.json --output outputs/suite
```

접촉 시나리오는 자유 낙하, 착지, 반발, 정착, 정지, sliding, 모서리 회전, 벽 단독, 바닥–벽 동시 접촉입니다. 학습에 포함하지 않은 동시 접촉 조합은 OOD split로 분리합니다. 데이터는 trajectory 단위로 나뉘며 접촉 사건 진단은 내부 substep에서도 기록합니다.

비교 모델: `state_mlp`, `nerd_like`, `deepsets`, `flat`, `independent`, `hierarchical`. **`nerd_like`는 내부 통제 비교군이며 원본 NeRD 재현이 아닙니다.** Hilbert/Morton, velocity·inertia 제거, mean pooling, 무작위 face grouping, face token 수, face 간 정보 교환, history 등의 설정은 `configs/suite_research.json`과 모델 문서를 참고하세요.

## 전이 학습: Cube → 직육면체 / 두 링크 / Ant

```bash
docker compose run --rm research python -m cube generate --config configs/two_link_smoke.json --output data/two_link
docker compose run --rm research python -m cube transfer --source-checkpoint outputs/cube/best.pt --target-dataset data/two_link --config configs/transfer_smoke.json --output outputs/transfer
docker compose run --rm research python -m cube generate --config configs/cuboid_transfer.json --output data/cuboid
docker compose run --rm research python -m cube generate --config configs/ant_smoke.json --output data/ant
docker compose run --rm research python -m cube train --dataset data/ant --config configs/train_smoke.json --transfer-checkpoint outputs/cube/best.pt --transfer-mode point-frozen --output outputs/ant_transfer
```

전이 비교는 `scratch`, `full`, `point-frozen`, `aggregation-frozen`입니다. 새로운 관절 decoder는 target에서 학습하며, 불러온 파라미터와 freeze 예외를 `training.json`에 기록합니다. 실제 target trajectory 수보다 큰 데이터 예산은 오류로 처리합니다.

## Ant 사족 보행과 모델 기반 제어

```bash
docker compose run --rm research python -m cube control --kind ant --controller baseline --steps 300 --output outputs/ant_walk
docker compose run --rm research python -m cube control --kind ant --controller zero --steps 300 --output outputs/ant_zero
docker compose run --rm research python -m cube train --dataset data/ant --config configs/train_smoke.json --output outputs/ant
docker compose run --rm research python -m cube evaluate --checkpoint outputs/ant/best.pt --dataset data/ant --output outputs/ant/test
docker compose run --rm research python -m cube control --kind ant --controller mpc --checkpoint outputs/ant/best.pt --steps 100 --horizon 3 --candidates 8 --iterations 1 --output outputs/ant_mpc
docker compose run --rm research python -m cube control --kind cube --controller mpc --checkpoint outputs/cube/best.pt --goal 0.3 0 0.2 --output outputs/cube_mpc
```

Ant는 4개 다리·8개 motor의 자체 모델입니다. baseline은 PD diagonal trot이고, MPC는 학습한 동역학으로 후보 action을 계획한 뒤 **독립적인 실제 simulator step**에서 평가합니다. RL 학습 정책이나 실제 하드웨어 검증을 의미하지 않습니다.

## Warp 데이터 생성과 GPU 학습

```bash
docker compose --profile gpu run --rm warp python -m cube generate --config configs/cube_smoke.json --backend warp --device cuda:0 --output data/cube_warp
docker compose --profile gpu run --rm warp python -m cube control --kind ant --controller baseline --backend warp --device cuda:0 --steps 300 --output outputs/ant_warp
```

`warp` 서비스는 실제 Warp CUDA physics와 CPU PyTorch를 사용합니다. 신경망도 CUDA로 학습하려면 `cuda` 이미지를 빌드하고 학습 JSON의 `device`를 `cuda:0`으로 설정합니다.

```bash
docker compose --profile cuda build cuda
docker compose --profile cuda run --rm cuda python -m cube doctor
docker compose --profile cuda run --rm cuda python -m cube train --dataset data/cube --config configs/train_cuda.json --output outputs/cube_cuda
```

Warp backend는 요청한 장치를 사용할 수 없으면 오류를 내며 다른 simulator로 대체하지 않습니다. 현재 adapter는 단일 world correctness 경로이며 대규모 GPU batching 속도를 주장하지 않습니다.

## 학습 GT 데이터와 모델 rollout 시각화

**[시각화 구성과 사용법](http://localhost:8000/14_visualization.html)**에 데이터 선택, GT 재생, checkpoint의 자율 rollout 비교와 결과 해석을 정리했습니다. 기존 `render`는 NPZ의 저장된 상태를 재생하는 명령입니다. 새 명령은 대규모 NPY shard도 직접 읽고, 선택한 trajectory의 영상·시계열 그래프·JSON 지표를 HTML 갤러리로 만듭니다.

먼저 학습 데이터의 trajectory 목록을 보고 GT 영상을 생성합니다. 시뮬레이션을 다시 실행하지 않고 저장된 정답 상태를 렌더링합니다.

```bash
docker compose run --rm research python -m cube list-trajectories --dataset data/cube-toss --split train --scenario toss --limit 10
docker compose run --rm -e MUJOCO_GL=osmesa research python -m cube visualize-data --dataset data/cube-toss --split train --scenario toss --limit 6 --output docs/visualizations/gt-toss
docker compose run --rm research python scripts/make_toss_preview.py
docker compose up -d docs
```

**[새 투척 GT 갤러리](http://localhost:8000/visualizations/gt-toss/index.html)**에서 확인합니다. 기존 데이터의 다른 움직임은 **[9개 시나리오 비교](http://localhost:8000/visualizations/gt-scenarios/index.html)**에서 볼 수 있습니다. `--scenario`를 생략하면 해당 split 전체에서 선택합니다. 목록의 특정 ID만 보려면 `--trajectory-id`를 사용합니다.

**[기존 모델 rollout 비교 갤러리](http://localhost:8000/visualizations/model-impact/index.html)**는 이전 데이터의 32 trajectories·3 epochs checkpoint 기록이며 새 투척 학습 결과가 아닙니다. 기존 원본 데이터 제거 후에도 만들어 둔 갤러리는 볼 수 있습니다.

이미 생성한 **새 데이터 16 trajectories·5 epochs 검증 모델**의 비교 영상은 다음 명령으로 재현합니다. 이 checkpoint는 메인 구조의 학습 동작 확인용이며 수렴된 모델이 아닙니다.

```bash
docker compose run --rm -e MUJOCO_GL=osmesa research python -m cube visualize-rollout --dataset data/cube-toss --checkpoint outputs/toss-validation/hierarchical/best.pt --split test --scenario toss --limit 2 --horizon 100 --device cpu --output docs/visualizations/model-toss-validation
```

새 투척 데이터의 5-epoch suite 학습을 완료한 뒤에는 아래 명령을 사용합니다. 왼쪽 GT와 오른쪽 모델 예측을 같은 시각·카메라에서 비교하며, 모델은 초기 GT 상태에서 자기 예측을 다음 입력으로 사용합니다. 새 Cube의 action은 항상 0입니다. `evaluate`에서 `save_rollouts: false`로 실행했어도 checkpoint와 원본 dataset만 있으면 시각화할 수 있습니다.

```bash
docker compose --profile cuda run --rm -e MUJOCO_GL=osmesa cuda python -m cube visualize-rollout --dataset data/cube-toss --checkpoint outputs/cube-toss-suite-5ep/hierarchical_budget10000_seed0/best.pt --split test --scenario toss --limit 6 --horizon 100 --device cuda:0 --output docs/visualizations/model-toss
```

CUDA 추론과 OOD 바닥–벽 동시 접촉 비교는 다음과 같습니다. checkpoint가 존재해야 실행할 수 있습니다.

```bash
docker compose --profile cuda run --rm -e MUJOCO_GL=osmesa cuda python -m cube visualize-rollout --dataset data/cube-toss --checkpoint outputs/cube-toss-suite-5ep/hierarchical_budget10000_seed0/best.pt --split ood --scenario floor_wall --limit 2 --horizon 100 --device cuda:0 --output docs/visualizations/model-toss-ood
docker compose run --rm research python -m cube docs
```

`--width 640 --height 480 --stride 2`로 영상 해상도와 저장 프레임 수를 조절할 수 있습니다. `--stride`는 영상만 줄이며 모델 rollout의 계산 간격은 바꾸지 않습니다. 시각화 추론에는 W&B 인증이 필요하지 않습니다.

## 저장된 NPZ 재생과 테스트

```bash
docker compose run --rm -e MUJOCO_GL=osmesa research python -m cube render --trajectory outputs/ant_walk/rollout.npz --scene outputs/ant_walk/scene.json --output docs/assets/ant-walking.mp4 --stride 2
docker compose run --rm research python -m pytest -q -m "not warp"
docker compose --profile gpu run --rm -e CUBE_TEST_WARP=1 -e CUBE_WARP_DEVICE=cuda:0 warp python -m pytest -q
```

영상은 저장된 상태를 렌더링하므로 시뮬레이션을 다시 수행하지 않습니다. 화면 없는 CPU 환경에서는 `MUJOCO_GL=osmesa`가 안정적입니다.

## 본 실험 설정

NeRD 규모의 주 비교에는 위의 `cube_toss_nerd_scale.json`, `train_gpu.json`, `suite_gpu.json`을 사용합니다. `cube_nerd_scale.json`은 8개 접촉 시나리오 진단용 설정으로 남기되 앞으로 생성하는 rigid action은 0으로 바뀌었습니다. 기존 `data/cube-nerd`를 새 버전으로 재개하지 않습니다. 기존 `*_research.json`은 별도의 ablation·중간 규모 설정입니다. 본 실험의 성공은 아직 보장되지 않으며, 짧은 실행 결과의 결론과 제한은 HTML 결과 문서에 구분해 기록했습니다.

```bash
docker compose run --rm research python -m cube generate --config configs/cube_research.json --output outputs/cube-research
docker compose run --rm research python -m cube suite --config configs/suite_research.json --output outputs/research
```

| 경로 | 구성 |
|---|---|
| `src/cube/sim/` | MJCF 생성, 상태 계약, MuJoCo/Warp 어댑터 |
| `src/cube/models/` | 미분 가능한 표면/FK, 직렬화, 계층형 모델과 비교군 |
| `src/cube/experiments/` | 데이터, 학습, 평가, 개입, 전이, MPC |
| `configs/` | smoke·research 실험 설정 |
| `tests/` | 물리 계약, FK/gradient, 데이터 분리, 학습 통합 검증 |
| `docs/` | localhost에서 보는 HTML 문서·그래프·영상 |
| `documentation/` | HTML 문서 생성용 원본 |
| `outputs/quick/` | 직접 실행한 실험의 체크포인트·JSON·NPZ |
