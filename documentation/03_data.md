# 데이터와 접촉 사건 기록

## 무엇을 구현했는가

`src/cube/experiments/data.py`는 공통 simulator에서 trajectory를 생성하고, `storage.py`는 저장과 읽기를 담당합니다. 큰 데이터는 여러 trajectory를 묶은 NPY shard와 `manifest.json`으로 저장합니다. 기존 작은 실험의 trajectory별 압축 NPZ도 계속 읽고 생성할 수 있습니다. Cube, 직육면체, 두 링크, Ant에 동일한 상태 계약을 적용합니다.

| 저장 항목 | 의미 | 입력에 사용하는가 |
|---|---|---|
| states | qpos와 qvel, T+1개 상태 | 현재 상태·허용된 history만 |
| actions | T개 action; 무구동 Cube는 6차원 모두 0, Ant는 관절 control | 사용; Cube의 0 배열은 공통 schema 유지용 |
| scene_spec | 형상, 질량, 마찰, solver 설정, seed | 현재 물성/기하 구성에 사용 |
| contact_count / penetration | 현재 endpoint 접촉과 침투 | 진단·평가에만 사용 |
| impact_event / persistent_contact | 내부 substep에서 관찰한 사건 | loss 입력이 아닌 평가 구간 분류 |
| first_impact_time | 관찰한 최초 충돌의 substep 시각 | 진단에만 사용 |

모든 포인트는 현재 state와 정적 기하에서 다시 생성합니다. 미래 접촉 mask나 solver 충격량을 network 입력에 넣지 않습니다.

## 왜 이렇게 나누는가

인접 transition을 무작위로 나누면 같은 궤적이 train과 test에 들어갑니다. 따라서 독립적인 seed·초기 조건의 trajectory 전체를 split에 할당합니다. 새로운 Cube `toss`는 위치·SO(3) 회전·선속도·각속도를 seed마다 독립적으로 표본 추출합니다. 정규화 통계는 train split에서만 계산합니다.

`rigid_action_mode: "zero"`가 Cube/직육면체 데이터의 기본 정책입니다. 투척은 초기 속도로 표현하며 이후 외력·토크 action을 주지 않습니다. `random_wrench`는 외부 힘이 있는 별도 진단 실험에서 명시적으로 선택할 수 있지만 `toss`에는 사용할 수 없습니다. 과거 `data/cube-nerd`에는 일부 시나리오의 random wrench가 있었습니다. 새 데이터 검증 후 기존 원본을 제거했으며, [투척 구현과 기존 데이터 감사](15_cube_toss.md)에 차이와 보존한 감사 결과를 기록합니다.

OOD의 바닥–벽 동시 접촉은 학습 시나리오에 포함하지 않습니다. 어떤 시나리오가 OOD에 들어가는지는 설정과 manifest에 명시합니다. 같은 시나리오가 훈련과 OOD 목록에 동시에 들어가면 오류를 냅니다.

## 내부 substep과 접촉 모드

제어 간격 0.02초 안에 0.005초짜리 물리 step 네 번을 수행합니다. endpoint만 관찰하면 그 사이에 충돌하고 다시 분리된 사건을 놓칠 수 있습니다. 데이터 생성은 각 substep의 현재 기하·접근 속도·활성 접촉을 기록하여 impact와 persistent 구간을 구분합니다. 한 transition에서 둘 다 발생할 수 있습니다.

접근 속도 기준은 −0.05m/s입니다. 접촉 쌍의 활성화와 접근 속도를 함께 사용하며 충격량 크기만으로 모드를 정의하지 않습니다. 현재 구현은 geom pair 단위 사건 진단입니다. 표면별 mode probe나 auxiliary supervision은 별도 연구 확장으로 남깁니다.

학습 모델의 장기 contact-onset 오차는 모델 출력 간격의 endpoint에서 계산합니다. 내부 substep의 정확한 발생 시점을 출력하는 모델인 것처럼 해석하지 않습니다.

## 큰 데이터의 저장·생성 구성

| 논리적 구성 요소 | 구현 내용 | 구현한 이유 |
|---|---|---|
| 생성 전 계획 | `data-plan`으로 split별 trajectory·transition 수, shard 수, 배열 용량을 계산 | 시나리오당 수량과 전체 수량을 혼동하지 않고 저장 공간을 확인 |
| 묶음 저장 | 최대 256 trajectories의 states/actions/labels를 각각 `.npy` 배열에 저장 | 작은 파일을 수만 개 열거나 데이터 전체를 한 번에 압축 해제하는 비용을 줄임 |
| 부분 읽기 | 읽기 전용 memory map과 제한된 파일 캐시, 필요한 배열만 요청 | 학습이 states/actions만 읽고 전체 데이터가 RAM·VRAM에 상주하지 않도록 함 |
| MuJoCo 병렬 생성 | subprocess worker 8개, worker 안에서 같은 장면의 컴파일된 모델을 재사용 | trajectory마다 MJCF를 다시 컴파일하는 비용을 줄이고 CPU 작업을 분산 |
| 독립 episode | trajectory ID로 정한 seed로 초기 상태·action을 다시 생성하고 simulator를 reset | worker 완료 순서가 달라도 동일 trajectory를 생성 |
| 완료 단위 저장 | 임시 shard의 모든 배열과 metadata를 기록한 뒤 원자적으로 완료 경로로 변경 | 중단된 파일을 정상 데이터로 읽지 않도록 함 |
| 데이터 식별 | 생성 설정·seed·backend·정밀도·저장 구조·라벨 관측 방식의 fingerprint와 schema hash | 다른 실험 데이터가 같은 경로에 섞이는 것을 방지 |

`configs/cube_toss_nerd_scale.json`의 `storage.format`은 `sharded_npy`, `trajectories_per_shard`는 256, 저장 `dtype`은 float32입니다. MuJoCo 적분은 simulator의 정밀도로 수행하고 저장할 때 변환합니다. `generation.workers: 8`은 CPU 생성 병렬도이며 GPU 학습의 batch 수와 관계없습니다. 작업 제출 수도 제한하여 전체 trajectory 배열을 부모 프로세스에 모으지 않습니다.

`generation.record_substeps: true`가 기본입니다. `false`는 제어 간격 양 끝만 관찰하는 별도 설정이며, 중간 충돌을 놓칠 수 있습니다. 이 경우 manifest의 시간 해상도도 endpoint 관측으로 바뀝니다. 본 실험에서는 정확한 사건 분류를 위해 substep 기록을 유지합니다.

Warp도 동일 데이터 형식으로 저장합니다. 현재 Warp adapter는 실제 CUDA 물리를 수행하는 **단일 world 경로**이므로 `generation.workers`를 1로 설정해야 합니다. 이 구현이 GPU 다중 world 생성의 처리량을 제공한다고 가정하지 않습니다.

## NeRD Cube 수량과 실제 생성 결과

[NeRD Appendix C.4](https://arxiv.org/html/2508.15755v1#A3.SS4)의 Cube synthetic 학습 데이터는 랜덤 초기 상태의 투척 10,000 trajectories, 길이 100입니다. 이 저장소에서는 길이를 **100 transitions와 101 states**로 정의합니다. 새 주 설정 `cube_toss_nerd_scale.json`은 `toss` 하나에 train 10,000개, validation/test 각 1,000개를 배정합니다. OOD는 기존 바닥–벽 동시 접촉 진단인 `floor_wall` 1,000개로, 랜덤 투척 OOD라고 부르지 않습니다. 초기 조건의 구체적인 표본 범위는 자체 선택이며 원논문의 solver·분포·실측 데이터까지 재현하는 것은 아닙니다.

새 `data/cube-toss` **13,000개 전체 생성을 181.42초에 완료**했습니다. 전수 검사에서 nonzero action과 비유한 상태는 0개였으며, 12,000개 toss의 초기 접촉은 0개, 고유 초기 상태는 12,000개였습니다. [새 데이터 감사 결과](15_cube_toss.md)에서 분포와 물리 정밀도의 제한까지 확인할 수 있습니다.

**과거 기록:** 2026-10-18에 생성한 `data/cube-nerd`도 총 13,000 trajectories / 1,300,000 transitions지만, 8개 접촉 시나리오에 train 각 1,250개, validation/test 각 125개를 배정한 데이터입니다. 생성 시간 약 172초, 배열 payload 약 128 MB는 이 기존 데이터의 측정값입니다. 일부 시나리오는 random wrench가 있었고, impact 초기 자세는 단위 회전으로 고정되어 있으므로 새 무구동 투척 데이터와 구별합니다. [이전 GPU 실행 결과](13_gpu_results.html)도 기존 데이터의 결과입니다. 새 데이터 생성과 확인 결과는 [투척 데이터 문서](15_cube_toss.md)에 기록합니다.

## 재현 정보와 안전한 재개

manifest와 shard metadata에는 seed, trajectory별 scene spec과 split, simulator backend, 시간 간격, 배열 shape/dtype을 남깁니다. `quick` 결과에는 설치 버전과 소스·설정 SHA256도 기록합니다.

같은 출력 경로에 다시 생성하려면 `--resume`를 명시합니다. 기본 명령은 기존 데이터를 덮어쓰지 않습니다. **미완료 데이터**는 이미 완료된 shard의 설정·trajectory ID·배열 shape/dtype·파일 크기를 검증하고 빠진 shard부터 생성합니다. 완료되지 않은 임시 shard는 학습 대상에 포함하지 않습니다. 부모 프로세스가 manifest를 갱신하기 직전에 중단됐더라도, 정상적으로 완료된 shard는 재개 시 다시 찾습니다. 학습·평가는 manifest가 `incomplete`이면 오류를 냅니다.

**완료된 데이터**의 `--resume`는 동일 생성 버전·설정의 기존 파일을 검증한 뒤 그대로 재사용하며 manifest 내용도 바꾸지 않습니다. 따라서 학습 checkpoint가 참조하는 데이터 식별자가 유지됩니다. seed, 물리 설정, backend, 저장 형식, 사건 기록 방식, 생성 정책 버전 등이 달라지면 재개를 거부하므로 새 출력 폴더를 사용합니다. 이번 action 정책과 초기 상태 변경은 생성 버전도 변경합니다. **기존 `data/cube-nerd`에 새 코드·설정으로 생성 재개하지 않고 `data/cube-toss`를 사용합니다.** 기존 원본 제거 후에는 보존한 감사 결과와 갤러리로 과거 분포를 확인합니다. worker 수와 진행률 표시 설정은 데이터 내용이 같으므로 변경할 수 있습니다. fingerprint가 없던 이전 NPZ 데이터는 읽을 수 있지만 이 방식으로 생성을 재개할 수는 없습니다.

```bash
docker compose run --rm research python -m cube data-plan --config configs/cube_toss_nerd_scale.json

docker compose run --rm research python -m cube generate --config configs/cube_toss_nerd_scale.json --output data/cube-toss --resume

docker compose run --rm research python -m cube generate --config configs/cube_toss_smoke.json --output data/cube-toss-small

docker compose --profile gpu run --rm warp python -m cube generate --config configs/cube_toss_smoke.json --backend warp --device cuda:0 --output data/cube-toss-warp-check
```

계획의 용량은 배열 payload의 비압축 크기이며 metadata와 임시 shard를 위한 여유 공간은 별도로 필요합니다. 전체 GPU 학습·평가 명령은 [GPU 실험 안내](12_gpu_training.html)를 참고하세요.
