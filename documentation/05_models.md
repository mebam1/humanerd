# 계층 동역학 모델·비교군·전이 구성

## 구현한 것과 이유

`src/cube/models/dynamics.py`는 공통 `DynamicsModel` 인터페이스에서 계층 모델, 비교군, ablation, 전이용 모듈 경계를 제공합니다. 같은 상태·행동·정적 형상에서 같은 generalized delta를 예측하게 하여, 데이터나 출력 표현 차이가 구조 비교에 섞이는 것을 줄이기 위한 구성입니다.

모델은 다음 상태나 접촉 충격량을 출력하는 solver가 아닙니다. 현재 표면의 국소 표현을 만든 뒤 여러 표면과 링크의 정보를 결합하여 **하나의 root 상태와 연결된 관절 상태 변화**를 예측합니다. Ant의 각 링크를 서로 독립적인 pose로 디코딩하지 않습니다.

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
| `nerd_like` | 현재 기하 기반 근접 표면 요약, 상태 문맥, history Transformer | 접촉 요약·history를 제공하는 자체 비교군은 어느 수준인가? |
| `deepsets` | 동일한 local point encoder 후 동적 포인트 mean과 비선형 readout | 단순 pooling과 비선형 decoder만으로 충분한가? |
| `flat` | face 계층 없이 전체 포인트 attention과 global readout | 물리적 grouping 없이 전역 attention만으로 충분한가? |
| `independent` | 각 정적 geometry의 국소 반응을 독립 계산한 뒤 선형 합산 | 서로 떨어진 지지의 반응을 독립적으로 더하는 가정이 충분한가? |

**`nerd_like`는 공식 NeRD 구현·재현 결과가 아닙니다.** 이 저장소의 현재 상태 표현과 출력 규약을 공유하는 자체 비교군입니다. 실제 NeRD, FIGNet, RigidFormer의 논문 성능을 재현했다고 주장하지 않습니다. 원본 구현, 학습 데이터, solver 설정을 정합한 별도 비교가 필요합니다.

`nerd_like`는 가까운 정적 표면의 위치·법선·거리·상대속도·재질 정보를 current geometry에서 계산해 요약합니다. solver가 계산한 미래 접촉이나 충격량은 사용하지 않습니다. `history=1`과 `history=10`을 지원합니다. history에는 현재 시점을 포함하며, rollout에서는 마지막 state를 실제 예측 state로 교체합니다. 상태와 행동 history를 함께 전달하면 과거 torque가 보존됩니다. 상태 history만 API에 전달한 경우에는 현재 행동을 반복하므로, 정식 비교에는 행동 history도 전달해야 합니다.

`state_mlp`에도 정적 환경 요약을 제공합니다. 지지대 유무를 모델에 숨긴 채 동일한 상태에 서로 다른 정답을 강요하는 비교를 피하기 위한 조치입니다. 따라서 순수한 `[state, action]`만 받는 MLP와는 다릅니다.

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

모든 형상별 샘플과 FK 상수는 `features`의 **nonpersistent buffer**입니다. checkpoint의 학습된 `state_dict`에 넣지 않습니다. 타깃 모델은 타깃 메타데이터로 새로 만든 후, 모양이 같은 학습 tensor만 옮깁니다. Cube의 형상 buffer가 Ant 모델을 덮어쓰는 문제를 막습니다.

`model.set_scene(sim_or_metadata)`는 DOF가 같은 환경 사이에서 정적 형상만 바꿀 때 사용합니다. learned parameter의 객체를 바꾸지 않으므로 optimizer가 계속 같은 parameter를 갱신합니다. DOF가 달라지면 새 모델을 만들도록 오류를 냅니다.

전이 모드는 처음부터 학습, 전체 fine-tuning, point encoder 고정, 상위 aggregation 고정입니다. `freeze_transfer('point')`는 point encoder만 고정합니다. `freeze_transfer('aggregation')`는 point encoder 이외의 학습 parameter를 고정합니다.

Cube에서 관절 시스템으로 처음 옮길 때는 joint decoder가 소스 학습에 참여하지 않았습니다. 이 경우 그 무작위 소스 head를 복사하지 않고 **새 타깃 joint head를 학습 가능하게 유지**합니다. aggregation 고정 실험도 이 예외를 적용하며, 결과는 “기존 aggregation 고정 + 새 joint head 학습”으로 해석해야 합니다. 같은 DOF 전이의 순수 고정 실험과 구분합니다.

## 공통 API와 검증

`build_model(sim_or_metadata, config)`가 모델을 생성합니다. `forward(state, action, history=None)`는 raw delta를 반환하며 학습 정규화는 trainer가 처리합니다. `encode_state`는 probe용 latent를, `coverage`는 local attention 근접쌍 포함 통계를 제공합니다. 실제 사용 명령은 README의 Docker Compose 명령을 따릅니다.

모델 검증에는 Cube·Ant의 모든 비교군 출력 크기, 두 번의 예측 상태 재구성을 포함한 유한 gradient, patch 내부 포인트 순서 변경에 대한 pooling 불변성, 모든 ablation 실행, 타깃 형상 보존, point encoder 고정, 새 joint head 학습 가능 여부가 포함됩니다.

이 구현은 실험 가능한 연구 기준 코드입니다. attention 사용만으로 비관통·에너지 보존·마찰 제약을 보장하지 않습니다. 짧은 smoke 학습의 오차 순위는 논문 가설을 입증하지 않습니다. Baseline별 parameter 수와 실제 활성 경로, 전체 전처리 포함 시간, 데이터 예산, 여러 seed를 함께 보고해야 합니다. 공통 클래스에는 특정 비교군에서 쓰지 않는 모듈이나 Cube에서 아직 활성화되지 않은 joint head가 있을 수 있으므로 총 parameter 수만으로 계산량이 같다고 판단하지 않습니다.
