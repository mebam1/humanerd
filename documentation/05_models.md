# 계층 동역학 모델·비교군·전이 구성

## 구현한 것과 이유

`src/cube/models/dynamics.py`는 계층 모델, point 기반 비교군, ablation, 전이용 모듈 경계를 제공합니다. `src/cube/models/contact.py`의 `contact_mlp`는 같은 예측 API를 사용하면서, NeRD의 명시적 접촉 표현을 독립적으로 구성합니다. 모델들은 같은 generalized delta를 출력하지만, 환경 입력을 만드는 방식과 입력 정규화는 비교군마다 다릅니다.

모델은 접촉 충격량 대신 **하나의 root 상태와 연결된 관절 상태 변화**를 예측하고, 이 변화로 다음 상태를 재구성합니다. Point 기반 모델은 현재 표면의 국소 표현을 만든 뒤 여러 표면과 링크의 정보를 결합합니다. Ant의 각 링크를 서로 독립적인 pose로 디코딩하지 않습니다. `contact_mlp`의 현재 구현 범위는 단일 자유 강체 Cube입니다.

## Point → Patch → Link → Articulation

| 논리적 구성 요소 | 구현한 동작 | 구현한 이유 |
|---|---|---|
| `point_encoder.input` | 21차원 물리 포인트를 공통 hidden feature로 변환 | 형상·포인트 수에 독립적인 하위 표현 생성 |
| `point_encoder.blocks` | 공동 Hilbert 정렬과 shifted local attention | 가까운 물체–환경 표면의 위치·속도·재질 상호작용 표현 |
| `aggregation.patch_query` | patch 중심·법선·extent로 query 생성 | 물리적 표면 소속과 상대 기하 유지 |
| `aggregation.face_attention` | 각 patch 소속 포인트에 query cross-attention | 면 내부의 여러 지지점을 함께 요약 |
| 링크 내부 face exchange | 같은 링크의 face token 사이 attention | 한 강체의 서로 다른 면에 걸친 접촉 결합 |
| `aggregation.link_attention` | 링크 물리 문맥에서 face token을 조회 | 접촉 표현에 질량·관성·동작 문맥 결합 |
| articulation attention | 전체 링크와 articulation token 사이 attention | 링크 연결과 다른 다리의 접촉 영향을 공동 추론 |
| base/joint decoder | root 12차원과 관절별 2차원 변화 | 유효한 연결 구조를 가진 generalized state 생성 |

Patch당 query offset은 학습 가능하며 `face_tokens=1,2,4` 중 선택합니다. Face 수에 비례하는 전용 embedding table을 만들지 않으므로, 다른 수의 면·링크로도 같은 point encoder와 상위 모듈을 옮길 수 있습니다.

관절 연결은 부모–자식, 자기 자신, 다른 링크, articulation token 관계로 나누어 학습 가능한 attention bias를 제공합니다. 관절축·관절 위치·속도·torque는 링크 입력에 포함됩니다. 기본 글로벌 attention은 모든 링크 사이의 교환을 허용합니다. 이 bias는 실제 질량행렬이나 제약 solver의 대체 계산을 의미하지 않습니다.

## 비교군의 정확한 범위

| 설정 이름 | 구현 내용 | 비교 질문 |
|---|---|---|
| `hierarchical` | Point→Patch→Link→Articulation | 물리적 계층이 유용한가? |
| `state_mlp` | 상태·행동·링크 문맥·정적 환경 요약을 MLP로 처리 | 전체 벡터 중심 표현은 어느 수준인가? |
| `contact_mlp` | Cube의 고정 접촉 후보점에서 계산한 명시적 접촉 정보와 body frame 운동 상태를 MLP로 처리 | NeRD 방식의 접촉 입력만으로 새로운 지지 형상에 대응할 수 있는가? |
| `nerd_like` | 현재 기하 기반 근접 표면 요약, 상태 문맥, history Transformer | 접촉 요약·history를 제공하는 자체 비교군은 어느 수준인가? |
| `deepsets` | 동일한 local point encoder 후 동적 포인트 mean과 비선형 readout | 단순 pooling과 비선형 decoder만으로 충분한가? |
| `flat` | face 계층 없이 전체 포인트 attention과 global readout | 물리적 grouping 없이 전역 attention만으로 충분한가? |
| `independent` | 각 정적 geometry의 국소 반응을 독립 계산한 뒤 선형 합산 | 서로 떨어진 지지의 반응을 독립적으로 더하는 가정이 충분한가? |

**`nerd_like`는 공식 NeRD 구현·재현 결과가 아닙니다.** 이 저장소의 point 기반 표면 표현과 출력 규약을 공유하는 기존 자체 비교군입니다. 아래의 `contact_mlp`와 입력 경로가 다릅니다. `contact_mlp` 역시 NeRD의 접촉 표현과 공개 MLP 구성을 적용한 Cube 기준모델이며, 공식 NeRD 벤치마크를 그대로 재현한 결과는 아닙니다. 실제 NeRD, FIGNet, RigidFormer의 논문 성능을 재현했다고 주장하려면 원본 구현, 학습 데이터, solver 설정을 정합한 별도 비교가 필요합니다.

`nerd_like`는 가까운 정적 표면의 위치·법선·거리·상대속도·재질 정보를 current geometry에서 계산해 요약합니다. solver가 계산한 미래 접촉이나 충격량은 사용하지 않습니다. `history=1`과 `history=10`을 지원합니다. history에는 현재 시점을 포함하며, rollout에서는 마지막 state를 실제 예측 state로 교체합니다. 상태와 행동 history를 함께 전달하면 과거 torque가 보존됩니다. 상태 history만 API에 전달한 경우에는 현재 행동을 반복하므로, 정식 비교에는 행동 history도 전달해야 합니다.

`state_mlp`에도 정적 환경 요약을 제공합니다. 지지대 유무를 모델에 숨긴 채 동일한 상태에 서로 다른 정답을 강요하는 비교를 피하기 위한 조치입니다. 따라서 순수한 `[state, action]`만 받는 MLP와는 다릅니다.

## `contact_mlp`: NeRD 방식의 명시적 접촉 입력

NeRD는 collision detection을 유지하고, 그 결과와 현재 운동 상태·작용 토크를 신경망에 입력하여 다음 상태 변화를 예측합니다. 논문의 접촉 표현은 후보점마다 `cᵢ = (p0ᵢ, p1ᵢ, nᵢ, dᵢ)`입니다. `p0`는 로봇 쪽 고정 후보점, `p1`은 환경 쪽 대응 표면점, `n`은 접촉 법선, `d`는 부호 있는 거리입니다. 접촉력·충격량이나 미래 GT contact는 입력하지 않습니다. [NeRD §3.2–3.3 및 Appendix A](https://arxiv.org/html/2508.15755v1)

공식 공개 코드에서 BOX는 꼭짓점 8개를 고정 후보점으로 사용합니다. `p0`는 collision query에 필요하지만 고정된 값이므로 공개 MLP의 신경망 입력 목록에서는 제외됩니다. `contact_mlp`도 이 구성을 따릅니다. 각 꼭짓점에 대해 현재 예측 상태와 장면 형상으로 접촉 레코드를 다시 계산하므로, rollout에 GT 접촉 정보를 공급하지 않습니다. [공식 후보점 생성 코드](https://github.com/NVlabs/neural-robot-dynamics/blob/370e89d2b8ce68aa3578aa7812859c681b7502bc/envs/abstract_contact_environment.py)

| 신경망 입력 순서 | 차원 | 표현 |
|---|---:|---|
| 현재 선속도·각속도 | 6 | Cube body frame |
| 후보점 8개의 법선 `n` | 24 | 환경에서 자유공간을 향하는 법선을 Cube body frame으로 변환 |
| 후보점 8개의 환경 표면점 `p1` | 24 | Cube 원점을 기준으로 옮기고 body frame으로 회전 |
| 후보점 8개의 거리 `d` | 8 | 분리 시 양수, 접촉 시 0, 침투 시 음수 |
| 외력·토크 | 6 | 현재 작용하는 wrench를 body frame으로 표현 |
| 단위 중력 방향 | 3 | Cube body frame |
| **합계** | **71** | 현재 step만 사용, `history=1` |

네트워크는 **`71 → 512 → 256 → 128 → 64 → 12`**이며, 은닉층마다 ReLU를 사용하고 LayerNorm은 사용하지 않습니다. 폭과 활성화는 공식 저장소의 공개 Ant MLP 설정을 따릅니다. 공식 main에는 Cube Toss 전용 MLP 학습 설정이 없으므로, 이를 Cube에 맞게 적용한 구성입니다. 출력 12차원은 현재 root frame에서의 위치 변화 3, 회전 벡터 3, 선속도 변화 3, 각속도 변화 3입니다. [공식 MLP 설정](https://github.com/NVlabs/neural-robot-dynamics/blob/370e89d2b8ce68aa3578aa7812859c681b7502bc/train/cfg/Ant/mlp.yaml), [공식 MLP 구성 코드](https://github.com/NVlabs/neural-robot-dynamics/blob/370e89d2b8ce68aa3578aa7812859c681b7502bc/models/models.py)

`ContactFeatures`는 평면과 회전된 box에 대한 해석적 point–surface query로 `p1`, `n`, `d`를 계산합니다. 환경 point cloud를 생성하지 않으며, `GeometryFeatures`, point encoder, point sampling, pooling을 사용하지 않습니다. 꼭짓점 순서는 x 부호가 가장 빨리 바뀌고, 다음 y, 다음 z인 고정 순서입니다. 정적 형상이 여러 개이면 각 꼭짓점에서 **signed distance가 가장 작은 표면 하나**를 선택합니다. 공식 공개 collision query가 단일 ground plane을 대상으로 하는 데서, 현재 저장소의 벽·V자 바닥으로 확장한 부분입니다.

`d > ξ`인 후보점의 전체 raw 접촉 레코드는 0으로 마스킹합니다. 선택된 geometry 쌍마다 `ξ = max(4 × max(Cube margin, 환경 geometry margin), 0.1 m)`로 정하고 경계 `d = ξ`는 유지합니다. 이는 논문의 0.1 m 기본값과 마스킹 조건을 채택한 것입니다. 공개 코드의 현재 상수는 0.12 m이고 `d < ξ`를 사용하므로, 이 차이를 구분해야 합니다. [공식 마스킹 함수](https://github.com/NVlabs/neural-robot-dynamics/blob/370e89d2b8ce68aa3578aa7812859c681b7502bc/integrators/integrator_neural.py), [공식 threshold 상수](https://github.com/NVlabs/neural-robot-dynamics/blob/370e89d2b8ce68aa3578aa7812859c681b7502bc/utils/commons.py)

정규화는 **body frame 변환 → raw contact 마스킹 → train split 통계로 입력 정규화** 순서입니다. 입력은 `(x − 평균) / sqrt(분산 + 1e−5)`로 변환합니다. 따라서 접촉이 없어 raw 값이 0이더라도 정규화된 NN 입력은 `−평균/std`일 수 있습니다. 네트워크 출력도 train split에서 얻은 변화량 통계로 정규화하여 학습하고, 추론 시 역정규화한 뒤 기존 `state_delta`·`apply_delta` 규약으로 다음 상태를 재구성합니다. 검증·test·OOD 데이터로 정규화 통계를 맞추지 않습니다. [공식 학습 전처리](https://github.com/NVlabs/neural-robot-dynamics/blob/370e89d2b8ce68aa3578aa7812859c681b7502bc/algorithms/vanilla_trainer.py), [공식 정규화 식](https://github.com/NVlabs/neural-robot-dynamics/blob/370e89d2b8ce68aa3578aa7812859c681b7502bc/utils/running_mean_std.py)

현재 구현의 범위와 NeRD 원본과의 차이는 다음과 같습니다.

- Cube의 자유 관절 속도는 MuJoCo 규약에 맞춰 처리합니다. 원문의 Warp spatial twist 변환식을 COM 선속도에 그대로 적용하지 않습니다.
- Cube에는 관절 토크 대신 외력·토크 6차원을 입력합니다. 위치·회전의 body frame 상수 성분은 NN 입력에서 생략합니다.
- 출력 회전은 저장소의 3차원 회전 벡터이며, 학습 손실은 저장소의 정규화 변화량 MSE 및 회전 항을 사용합니다. 공식 설정의 quaternion 출력과 동일한 실험은 아닙니다.
- 후보점 하나당 표면 하나만 표현하므로, 바닥·벽 모서리처럼 같은 꼭짓점에서 여러 표면과 동시에 접촉하는 경우를 모두 담을 수 없습니다. 꼭짓점 8개로 포착되지 않는 임의 형상의 접촉에도 한계가 있습니다.
- 질량·관성·마찰계수를 별도 입력하지 않는 Cube 전용 모델입니다. 다른 물체나 물성이 바뀌는 문제로의 일반화를 이 구성만으로 주장하지 않습니다.

학습 설정은 `configs/train_contact_mlp.json`, 별도 비교 실행 설정은 `configs/suite_contact_mlp.json`입니다. 두 설정의 W&B 로깅은 기본적으로 꺼져 있습니다. `outputs/contact-mlp-5ep/best.pt`는 10,000개 trajectory로 5 epoch·seed 0 학습을 완료했고, 기존 세 모델과 동일한 OOD v2의 1,000개·100스텝 조건으로 평가했습니다. 전체·벽·V자 바닥의 지표와 접촉 진단은 [OOD 비교 문서](../docs/ood_comparison.html)에서 확인할 수 있습니다.

## 독립 합산 기준선의 의미

`independent`는 동적 포인트와 각 정적 geometry의 포인트 쌍에 대해 공유 MLP 반응과 거리 gate를 계산합니다. 같은 geometry 안에서 평균을 내고, geometry별 기여를 합산한 뒤 동일한 자유 운동 응답을 더합니다. **합산 뒤 비선형 readout을 두지 않습니다.**

가장 가까운 지지점 하나만 선택하면 B를 추가할 때 A의 입력이 바뀌어, 의도하지 않은 비가산 효과가 생길 수 있습니다. 이 구현은 geometry마다 독립 계산하여 그 경쟁을 제거했습니다. 동일 상태·동일 A/B 형상을 사용한 네 반사실 조건의 모델 delta 결합량 `AB−A−B+none`이 수치 오차 수준인지 테스트합니다.

Delta의 각속도는 현재 root frame 기준이므로 world twist로 변환한 속도 결합량도 같은 독립성 검증에 사용할 수 있습니다. 단, 위치·회전 전체를 비선형 좌표로 비교하는 지표와 속도 결합량을 혼동하지 않아야 합니다.

반면 mean/sum pooling 뒤에 비선형 decoder가 있는 `deepsets`는 접촉 결합을 학습할 수 있습니다. 이를 독립 합산 기준선과 동일하게 취급하지 않습니다.

## Ablation

| 설정 | 제거·변경하는 부분 | 유지되는 부분 |
|---|---|---|
| `no_velocity=true` | 명시적 **포인트** 표면 속도를 0으로 변경 | 현재 상태와 body 속도 입력 |
| `no_inertia=true` | body 질량과 관성 입력을 0으로 변경 | 형상, 상태, 중력, 행동 |
| `mean_pool=true` | patch query attention을 mean pooling으로 변경 | patch 기하 query와 상위 집약 |
| `random_groups=true` | 고정 seed로 동적 포인트의 patch 소속을 재배치 | 점의 물리적 값과 총 patch 수 |
| `no_face_exchange=true` | 링크 내부 face-to-face attention 제거 | face→link 조회와 link→articulation 집약 |
| `serialization=morton` | Hilbert 대신 Morton 정렬 | 동일한 입력·window 구조 |
| `face_tokens=2/4` | patch별 query 수 증가 | 동일한 물리적 면 소속 |

`no_face_exchange`는 모든 접촉 결합을 차단하지 않습니다. 여러 face를 조회하는 link token은 여전히 결합을 표현할 수 있습니다. 각 ablation이 실제로 제거한 경로의 범위를 해석에 반영해야 합니다.

## 전이 학습 경계

Point 기반 모델의 형상별 샘플과 FK 상수는 `features`의 **nonpersistent buffer**입니다. checkpoint의 학습된 `state_dict`에 넣지 않습니다. 타깃 모델은 타깃 메타데이터로 새로 만든 후, 모양이 같은 학습 tensor만 옮깁니다. Cube의 형상 buffer가 Ant 모델을 덮어쓰는 문제를 막습니다. 아래의 point encoder·aggregation 전이 규약은 point 기반 모델에 해당하며, Cube 전용 `contact_mlp`의 Ant 전이를 의미하지 않습니다.

`model.set_scene(sim_or_metadata)`는 DOF가 같은 환경 사이에서 정적 형상만 바꿀 때 사용합니다. learned parameter의 객체를 바꾸지 않으므로 optimizer가 계속 같은 parameter를 갱신합니다. DOF가 달라지면 새 모델을 만들도록 오류를 냅니다.

전이 모드는 처음부터 학습, 전체 fine-tuning, point encoder 고정, 상위 aggregation 고정입니다. `freeze_transfer('point')`는 point encoder만 고정합니다. `freeze_transfer('aggregation')`는 point encoder 이외의 학습 parameter를 고정합니다.

Cube에서 관절 시스템으로 처음 옮길 때는 joint decoder가 소스 학습에 참여하지 않았습니다. 이 경우 그 무작위 소스 head를 복사하지 않고 **새 타깃 joint head를 학습 가능하게 유지**합니다. aggregation 고정 실험도 이 예외를 적용하며, 결과는 “기존 aggregation 고정 + 새 joint head 학습”으로 해석해야 합니다. 같은 DOF 전이의 순수 고정 실험과 구분합니다.

## 공통 API와 검증

`build_model(sim_or_metadata, config)`가 모델을 생성합니다. `forward(state, action, history=None)`는 역정규화된 raw delta를 반환합니다. Point 기반 모델의 기존 학습 정규화와 달리 `contact_mlp`에는 모델에 저장하는 train 입력·출력 통계가 필요합니다. `encode_state`는 probe용 latent를 반환합니다. `coverage`는 point 기반 local attention의 근접쌍 포함 통계이므로, point attention이 없는 `contact_mlp`에는 **해당 없음**으로 해석해야 합니다. 0이나 100%로 간주하여 point 모델과 비교하지 않습니다. 실제 사용 명령은 README의 Docker Compose 명령을 따릅니다.

Point 기반 모델 검증에는 Cube·Ant의 비교군 출력 크기, 두 번의 예측 상태 재구성을 포함한 유한 gradient, patch 내부 포인트 순서 변경에 대한 pooling 불변성, 모든 ablation 실행, 타깃 형상 보존, point encoder 고정, 새 joint head 학습 가능 여부가 포함됩니다. `contact_mlp`에는 그 대신 Cube 접촉 query, body frame 표현, 마스킹·정규화, 예측 상태에서의 접촉 재계산과 checkpoint 복원 검증이 필요합니다.

이 구현은 실험 가능한 연구 기준 코드입니다. attention 사용만으로 비관통·에너지 보존·마찰 제약을 보장하지 않습니다. 짧은 smoke 학습의 오차 순위는 논문 가설을 입증하지 않습니다. Baseline별 parameter 수와 실제 활성 경로, 전체 전처리 포함 시간, 데이터 예산, 여러 seed를 함께 보고해야 합니다. 공통 클래스에는 특정 비교군에서 쓰지 않는 모듈이나 Cube에서 아직 활성화되지 않은 joint head가 있을 수 있으므로 총 parameter 수만으로 계산량이 같다고 판단하지 않습니다.
