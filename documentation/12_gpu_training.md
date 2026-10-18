# GPU 학습 · NeRD 규모 데이터 · 온라인 실험 비교

기존 smoke 결과는 파이프라인 검증이다. 이 문서는 Cube의 데이터 양과 학습 예산을 늘려 **계층형 구조의 one-step 정확도, 자율 rollout 안정성, 접촉 조합 일반화**를 비교하는 실행 절차를 설명한다. 기본 GPU 실험은 7개 모델 설정 × 3개 seed = **21회 학습**이다. 설정을 준비했다는 사실과 장기 학습이 완료됐다는 사실은 구별해야 한다.

## NeRD에서 실제로 사용한 Cube 데이터 양

[Neural Robot Dynamics 원논문](https://arxiv.org/html/2508.15755v1)의 **Appendix C.4**는 Cube tossing의 synthetic pretraining에 **10,000 trajectories, 길이 100**을 사용했다고 명시한다. §4의 일반 로봇 설정은 **100,000 trajectories, 길이 100**이므로 Cube 전용 수치와 구별했다. C.4의 실측 데이터는 별도로 570 trajectories·약 60K transitions이며 train/validation/test는 400/85/85이다. 이 저장소는 해당 실측 데이터를 포함하지 않는다.

Appendix B의 공통 학습 설정은 batch 512, history 10, learning rate 1e-3 → 1e-4 선형 감소다. 논문에 고정된 총 epoch 수는 제시되지 않는다. 아래 5 epochs, 모델 구조, held-out split, 시나리오 비율은 이 연구의 비교 프로토콜이다. 원논문의 Warp Featherstone solver와 이 저장소의 MuJoCo/MuJoCo Warp solver도 다르다. **맞추는 것은 Cube의 synthetic 학습 데이터 양이며, NeRD의 전체 재현이라고 부르지 않는다.**

| 항목 | 기본 Cube 설정 | 용도 |
|---|---:|---|
| Training | 8 시나리오 × 1,250 = **10,000** | 정규화와 가중치 학습 |
| Validation | 8 × 125 = **1,000** | 독립 trajectory, checkpoint 선택 |
| Test | 8 × 125 = **1,000** | 최종 정확도·안정성 평가 |
| OOD | floor_wall **1,000** | 학습에 없는 바닥–벽 동시 접촉 |
| Trajectory 길이 | **100 transitions / 101 states** | 이 저장소의 `steps` 정의 |
| 전체 데이터 | **13,000 trajectories / 1,300,000 transitions** | train은 정확히 1,000,000 transitions |
| 시간 간격 | control 0.02 s, substep 0.005 s | trajectory당 2 s, 접촉 라벨은 substep 관측 |

설정의 `trajectories_per_split`은 **시나리오마다 적용되는 수**다. `train: 10000`을 넣으면 8배인 80,000개가 만들어지므로 기본값은 1,250이다. 모든 split은 독립 초기 조건·seed로 생성되며 같은 trajectory를 프레임 단위로 나누지 않는다. 100이라는 논문의 길이를 여기서는 저장된 상태 수가 아니라 transition 수로 해석했음을 config의 `reference`에도 남긴다.

추가 데이터 규모 실험에는 `configs/cube_nerd_general_scale.json`을 사용한다. 이는 §4의 일반 설정을 따른 train 100,000개·10,000,000 transitions이며, Cube 전용 기본 설정보다 10배 큰 별도 실험이다. validation/test/OOD는 각 1,000개이고 데이터 출력 폴더도 별도로 지정한다.

## 구현한 구성 요소와 이유

| 구성 요소 | 구현 내용 | 필요한 이유 |
|---|---|---|
| 데이터 계획 | `data-plan`이 split별 trajectory/transition 수와 저장 용량 추정치를 표시 | 수량 해석 오류와 생성 전 용량 부족을 확인 |
| Shard 저장 | `sharded_npy`, shard당 최대 256 trajectories, float32, manifest 인덱스 | 수많은 작은 NPZ 파일 대신 연속 배열을 memory map으로 읽기 |
| 생성 병렬화·재개 | MuJoCo worker 8개, 완료된 shard를 보존하는 `generate --resume` | 긴 CPU 생성 작업을 분산하고 중단 후 이어서 진행 |
| 제한된 학습 캐시 | trajectory·scene 캐시와 청크 단위 정규화 통계 | 전체 데이터·모든 MuJoCo 장면을 GPU/메모리에 올리지 않기 |
| GPU mixed precision | bfloat16 연산, gradient accumulation, gradient clipping | 작은 VRAM에서도 큰 nominal batch로 비교 |
| 학습 재개 | `best.pt`와 optimizer·scheduler·진행 위치를 담는 `last.pt` | 최적 모델 평가와 중단된 학습 재개의 역할을 구분 |
| tqdm | 생성·정규화·학습·평가 진행률, 속도, loss 표시 | 긴 작업에서 살아 있는 진행 상태와 대략적인 남은 시간을 확인 |
| W&B online | 설정, 학습·검증 지표, 학습 시간·속도와 실행 식별자 기록 | seed·구조별 learning curve를 같은 실험 그룹에서 비교 |
| Suite | 동일 dataset·budget·seed·학습 설정, test/OOD 평가, resume | 모델마다 다른 데이터나 학습량으로 인한 혼동을 줄이기 |

저장소 전체가 컨테이너 `/workspace`에 mount된다. 데이터·checkpoint·W&B 로컬 기록·HTML은 호스트의 `data/`, `outputs/`, `docs/`에 남는다. 기존 작은 NPZ 데이터도 읽을 수 있다.

## GPU와 W&B 준비

Docker Desktop Linux containers와 NVIDIA GPU를 준비한다. 저장소 루트에서 `.env.example`의 항목을 참고해 `.env`에 본인의 `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`를 채운다. `WANDB_ENTITY`에는 사용자 계정 또는 팀 이름을 넣는다. 키는 config JSON·명령행·문서에 넣지 않는다. Compose가 `.env`를 읽어 컨테이너에 전달한다.

```bash
docker compose build research
docker compose --profile cuda build cuda
docker compose --profile cuda run --rm cuda python -m cube doctor
```

`torch_cuda: true`와 사용할 GPU 이름을 확인한다. online 기록에는 W&B 계정 인증과 네트워크가 필요하다. 본 학습 config는 `wandb.mode: online`이며, 인증이 없거나 연결에 실패한 상태를 online 성공으로 간주하지 않는다. offline으로 동작만 확인할 때는 train/suite 명령 뒤에 `--wandb-mode offline`을 붙인다. 실제 계정에 성공적으로 run이 생성되고 지표가 보이는지 확인한 이후 장기 작업을 시작한다.

`train_gpu.json`의 group/tags는 실험 분류용이다. project는 기본적으로 `.env`의 `WANDB_PROJECT`를 사용하고, 값이 없으면 `cube-contact`다. config의 `wandb.project` 또는 단독 train CLI의 `--wandb-project`를 명시하면 해당 값이 환경변수보다 우선한다. `--wandb-entity`도 단독 train의 명시적 override다. 공통 비교에서는 동일 project와 group을 유지하고 모델·seed로 run을 구별한다. online 학습은 지표·허용된 실험 설정을 W&B로 보내며 API key 자체를 기록하지 않는다.

## 1. 데이터 수량을 확인하고 생성

```bash
docker compose run --rm research python -m cube data-plan --config configs/cube_nerd_scale.json
docker compose run --rm research python -m cube generate --config configs/cube_nerd_scale.json --backend mujoco --output data/cube-nerd
```

생성이 중단되면 **동일 config·backend·출력 폴더**로 이어서 진행한다.

```bash
docker compose run --rm research python -m cube generate --config configs/cube_nerd_scale.json --backend mujoco --output data/cube-nerd --resume
```

학습을 시작하기 전에 `data/cube-nerd/manifest.json`의 완성 상태와 train 10,000개를 확인한다. 완료되지 않은 데이터를 full dataset으로 취급하면 안 된다. 시뮬레이터와 신경망 학습 장치는 독립적이므로 MuJoCo CPU에서 만든 정답으로 CUDA 학습을 수행한다.

현재 작업에서는 이 기본 데이터 **13,000 trajectories, 1,300,000 transitions**의 생성을 직접 완료했다. MuJoCo worker 8개 실행은 **171.794초**였고, `data/cube-nerd/manifest.json`은 `status: complete`다. 기존 데이터가 있는 같은 작업 폴더에서는 생성 명령에 `--resume`을 붙여 완료된 shard를 재사용한다. 이 시간은 이 환경에서의 데이터 생성 측정이며 GPU 학습 소요 시간이 아니다.

Warp 물리 경로는 기존 validation 또는 작은 별도 데이터로 확인할 수 있다.

```bash
docker compose --profile gpu run --rm warp python -m cube validate --backend warp --device cuda:0 --output outputs/validation-warp
docker compose --profile gpu run --rm warp python -m cube generate --config configs/cube_smoke.json --backend warp --device cuda:0 --output data/cube-warp-check
```

현재 Warp adapter는 단일 world의 실제 CUDA physics 경로다. GPU 병렬 world 생성기의 속도를 제공한다고 가정하지 않는다. 큰 config를 Warp에 사용하려면 `generation.workers`를 1로 둔 별도 config와 출력 폴더를 사용한다. MuJoCo와 Warp 데이터를 섞는 실험은 먼저 backend 차이를 검증하고 별도로 기록한다.

## 2. 계층형 모델을 먼저 학습

```bash
docker compose --profile cuda run --rm cuda python -m cube train --dataset data/cube-nerd --config configs/train_gpu.json --model hierarchical --seed 0 --output outputs/cube-gpu-5ep/hierarchical
```

| 설정 | 기본값 | 해석 |
|---|---:|---|
| Epochs | 5 | 각 모델·seed가 전체 train transitions를 5회 학습 |
| Max optimizer steps | 없음 (`null`) | 별도 step 상한 없이 5 epochs에서 종료 |
| Microbatch / accumulation | 32 / 16 | nominal effective batch 512, 끝의 작은 batch는 실제 표본 수로 정규화 |
| Precision | bf16 | GPU가 지원해야 함 |
| Gradient norm clipping | 1.0 (global L2 norm) | AMP unscale·누적 gradient 평균화 후, optimizer step 전에 적용 |
| Learning rate | 0.001 → 0.0001 | 선형 감소, warmup 없음 |
| Rollout training | 1 step | 먼저 순수 one-step 학습으로 구조 비교 |
| Model | width 128, heads 4, local/global layers 3/3 | 모든 설정에서 공유 가능한 기본 구조 |
| Surface sampling | cube 면당 64, floor 256 points | smoke보다 조밀한 입력 |
| Validation selection | 독립 val 중 고정 256 trajectories | 매 epoch 비용을 줄이며 동일 subset 사용 |
| Checkpoint 간격 | optimizer 250 steps + epoch 종료 | `last.pt`로 재개, `best.pt`로 평가 |

5 epochs는 수렴을 보증하는 숫자가 아니다. W&B의 training/validation 추세와 실제 throughput을 보고 남은 시간을 판단한다. test/OOD 결과를 보고 hyperparameter를 계속 바꾸면 test가 tuning에 사용되므로, 연장·수정 여부는 validation으로 정한다. 비교할 모델 모두에 같은 변경 규칙을 적용하고 추가 실험 그룹으로 기록한다.

`--resume`는 scheduler의 전체 길이를 포함한 동일 최적화 설정을 확인하므로 `epochs`를 바꾸는 용도로 사용하지 않는다. 이전 50-epoch 실행의 checkpoint를 5-epoch 설정으로 재개하지 않고, `outputs/cube-gpu-suite-5ep`처럼 새 출력 폴더에서 비교를 시작한다. 기본 `max_steps`는 `null`이며 각 모델·seed는 5 epochs를 마치면 종료한다. 선택적 rollout fine-tuning도 실행당 5 epochs이며 별도 학습 단계다.

W&B의 `train/gradient_norm`은 `clip_grad_norm_`이 반환하는 **clipping 전 global norm**이다. 로그 값이 1.0보다 커도 clipping이 빠진 것이 아니다. 실제 optimizer에는 L2 norm이 1.0 이하로 제한된 gradient가 전달된다.

중단된 동일 실험은 아래처럼 재개한다. output과 config를 유지하며 `best.pt`가 아니라 `last.pt`를 사용한다.

```bash
docker compose --profile cuda run --rm cuda python -m cube train --dataset data/cube-nerd --config configs/train_gpu.json --model hierarchical --seed 0 --output outputs/cube-gpu-5ep/hierarchical --resume outputs/cube-gpu-5ep/hierarchical/last.pt
```

VRAM이 부족하면 microbatch를 줄이고 accumulation을 반비례로 늘려 nominal batch를 유지한다. 예를 들어 32×16에서 16×32 또는 8×64로 바꾼다. 입력 point 수나 hidden width를 특정 모델만 줄이면 다른 연구 질문이 되므로, 메모리 조절과 구조 축소를 구별한다.

학습의 bf16 batch 32와 평가는 별도로 설정한다. 본 suite의 **float32 평가 batch는 8**이다. wall_only 학습 장면은 768 points지만 floor_wall OOD는 바닥을 추가해 1,024 points가 되며, flat attention의 메모리는 point 수의 제곱에 비례한다. 실제 GPU check에서 float32 평가 batch 32의 OOD 처리량이 크게 낮아져 본 평가 batch를 줄였다. 입력 point 수와 평가 대상은 그대로 유지한다.

### 실제 GPU 용량 확인

RTX 4060, 총 VRAM 8,188 MiB에서 면당 64·floor 256 points, width 128, 3/3 layers, `wall_only` 장면으로 7개 모델의 짧은 forward/backward를 실행했다. microbatch 32가 모두 통과해 기본값을 32×16으로 정했다. 아래 수치는 이 짧은 capacity probe의 측정이며 수렴된 정확도나 전체 데이터 파이프라인의 처리량을 뜻하지 않는다. 메모리는 PyTorch peak allocated memory로, GPU 프로세스 전체 사용량과 다르다.

| 모델 | 짧은 probe 처리량 (samples/s) | Peak allocated (MiB) |
|---|---:|---:|
| hierarchical | 433.64 | 654 |
| state_mlp | 1,465.77 | 22 |
| deepsets | 567.66 | 650 |
| flat | 126.08 | 4,206 |
| independent | 243.06 | 3,292 |
| nerd_history1 | 1,365.13 | 47 |
| nerd_history10 | 437.12 | 52 |

원시 측정은 `outputs/runtime/gpu_capacity_b32.json`에 저장한다. 비교 모델의 계산량 차이가 크므로 총 wall time을 단일 모델의 속도로 추산하지 않는다. validation·자율 rollout·데이터 I/O가 실제 전체 실행에 추가된다. 4-step fine-tuning은 activation을 더 오래 유지하므로 별도의 작은 microbatch 설정을 사용한다.

## 3. 구조 비교와 seed 반복

```bash
docker compose --profile cuda run --rm cuda python -m cube suite --config configs/suite_gpu.json --output outputs/cube-gpu-suite-5ep
```

| Label | 비교하는 요소 |
|---|---|
| `hierarchical` | Point → Face → Body의 학습 가능한 aggregation |
| `state_mlp` | 상태·환경 요약의 작은 MLP 기준선 |
| `deepsets` | 같은 local point encoder 이후 평균 pooling을 사용하는 대조군 |
| `flat` | face/body 계층 대신 모든 point의 전역 attention |
| `independent` | 정적 형상별 반응을 독립적으로 합산하는 대조군 |
| `nerd_history1` | 명시적 접촉 요약과 현재 상태를 사용하는 NeRD-like |
| `nerd_history10` | 동일한 NeRD-like 구조에 길이 10의 history |

NeRD-like의 입력 요약, architecture, solver는 원본 NeRD와 다르다. 여기서 측정하는 것은 저장소 내부의 통제 비교다. 파라미터 수·wall time·VRAM 사용량은 구조마다 다르므로 데이터와 최적화 예산이 같다고 계산 비용까지 같다고 가정하지 않는다.

Suite는 `configs/train_gpu.json`을 읽고 동일 training 설정을 적용한다. 실행 경로는 `outputs/cube-gpu-suite-5ep/<label>_budget10000_seed<seed>/`다. 완성된 run의 결과와 seed별 요약은 `suite.json`에 기록한다. 기본 자동 평가는 test/OOD 각각 고정 최대 512 trajectories, 100-step rollout이며 원시 rollout 파일을 대량 저장하지 않는다. 전체 test 평가를 별도로 할 때는 checkpoint에 저장된 512 제한을 덮어쓰도록 아래 명령처럼 `--max-trajectories 1000`을 명시한다.

긴 비교를 중단했거나 일부 모델부터 확인했다면 이어서 진행한다.

```bash
docker compose --profile cuda run --rm cuda python -m cube suite --config configs/suite_gpu.json --output outputs/cube-gpu-suite-5ep --models hierarchical flat nerd_history10 --resume
docker compose --profile cuda run --rm cuda python -m cube suite --config configs/suite_gpu.json --output outputs/cube-gpu-suite-5ep --resume
```

완료한 run은 재사용하고 진행 중인 run은 checkpoint에서 이어간다. 모델 filter는 학습 범위를 선택하며 seed나 학습량을 자동 축소하지 않는다. 특정 모델 몇 개만 실행한 결과를 21회 전체 비교가 완료됐다고 해석하지 않는다.

## 4. 최종 test, OOD와 접촉 결합 실험

```bash
docker compose --profile cuda run --rm cuda python -m cube evaluate --checkpoint outputs/cube-gpu-5ep/hierarchical/best.pt --dataset data/cube-nerd --device cuda:0 --batch-size 8 --max-trajectories 1000 --rollout-horizon 100 --no-save-rollouts --output outputs/cube-gpu-5ep/hierarchical/test-full
docker compose --profile cuda run --rm cuda python -m cube evaluate --checkpoint outputs/cube-gpu-5ep/hierarchical/best.pt --dataset data/cube-nerd --split ood --device cuda:0 --batch-size 8 --max-trajectories 1000 --rollout-horizon 100 --no-save-rollouts --output outputs/cube-gpu-5ep/hierarchical/ood-full
docker compose run --rm research python -m cube coupling --checkpoint outputs/cube-gpu-5ep/hierarchical/best.pt --output outputs/cube-gpu-5ep/hierarchical/coupling
```

비교할 지표는 one-step 위치·회전·속도 오차, 100-step 자율 rollout의 오차와 발산율, 접촉 진입·유지 구간별 오차, OOD floor_wall 오차, A/B 접촉 개입의 결합량 예측 오차다. 평균 오차가 작아도 드문 충돌에서 발산하면 안정적인 simulator라고 결론낼 수 없다. 3 seeds의 평균과 표준편차, 실제 runtime을 함께 보고한다.

Coupling은 동일 초기 상태에서 A, B, A+B, 접촉 없는 조건을 비교한다. 학습에 없는 접촉 조합을 계층형 모델이 독립 합산보다 잘 예측하는지가 제안의 핵심 질문이다. 이 개입 실험은 기본 training loss와 분리된 평가다.

## 5. 선택 사항: 동일한 4-step rollout fine-tuning

기본 one-step 비교를 먼저 저장한 다음 `configs/train_gpu_rollout.json`으로 multi-step 손실의 효과를 분리해 본다. 예측 상태에서 geometry와 다음 입력을 다시 만들어 4-step rollout 전체에 gradient를 전파한다.

```bash
docker compose --profile cuda run --rm cuda python -m cube train --dataset data/cube-nerd --config configs/train_gpu_rollout.json --model hierarchical --seed 0 --transfer-checkpoint outputs/cube-gpu-5ep/hierarchical/best.pt --transfer-mode full --output outputs/cube-gpu-5ep/hierarchical-rollout4
```

이는 optimizer를 복원하는 resume와 달리, one-step 가중치에서 새 fine-tuning을 시작한다. config는 5 epochs·별도 optimizer step 상한 없음, microbatch 2 × accumulation 256, LR 1e-4 → 1e-5다. 성능을 비교하려면 모든 모델·seed에 같은 fine-tuning 예산과 평가를 적용한다. `nerd_history10`에는 기존 history 10을 유지한 별도 설정이 필요하다. 계층형만 fine-tune한 결과를 다른 모델의 one-step 결과와 섞어 순위를 만들지 않는다.

## HTML 확인과 실행 상태

```bash
docker compose run --rm research python -m cube docs
docker compose up -d docs
```

[localhost GPU 실험 안내](http://localhost:8000/12_gpu_training.html)에서 이 절차를 볼 수 있다. 10K train을 포함한 전체 데이터는 생성 완료했으며, 짧은 GPU 용량 검사는 실행 경로·메모리 확인용이다. 21회 장기 학습과 online 계정에서의 run 확인은 각각 별도의 산출물로 완료 여부를 판단한다. 실제 추가 측정과 제한은 [새 GPU 실행 결과](13_gpu_results.md)에 기록한다. 기존 [소규모 측정 결과](10_results.md)를 이번 장기 GPU 실험 결과로 대체하거나 재해석하지 않는다.
