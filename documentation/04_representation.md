# 현재 상태에서 만드는 물리 표면 표현

## 구현한 것과 이유

`src/cube/models/state.py`는 MuJoCo 상태와 학습 출력 사이의 변환을, `geometry.py`는 표면 포인트와 링크 문맥을, `serialization.py`는 공간 정렬과 근접쌍 포함률을 담당합니다. 시뮬레이터가 계산한 다음 상태나 충격량을 가져오지 않고, **현재 상태와 공개된 정적 형상 정보만으로 모델 입력을 재구성**하기 위해 분리했습니다.

이 구분은 rollout에서 특히 중요합니다. `model(predicted_state, action)`을 다시 호출하면 예측한 quaternion·관절각·속도에서 링크 변환, 표면 위치·속도, 환경 샘플 위치를 다시 계산합니다. 정답 포즈로 만든 포인트나 접촉 mask를 다음 단계에 공급하지 않습니다.

## 상태와 좌표계

지원 상태는 하나의 free root와 그 뒤의 hinge 관절들로 구성됩니다. `J`개 관절이면 `nq=7+J`, `nv=6+J`이고 상태 벡터는 `[qpos, qvel]` 순서의 `13+2J`차원입니다.

| 성분 | 저장 기준 |
|---|---|
| root 위치 | 월드 좌표계의 body 원점 |
| root quaternion | `wxyz`, body에서 world로 보내는 회전 |
| root 선속도 | body 원점의 world-frame 속도 |
| root 각속도 | 그 시점의 root body-frame 각속도 |
| 관절 | MuJoCo hinge 위치·속도 순서 |

각속도의 기준은 [MuJoCo floating-object 설명](https://mujoco.readthedocs.io/en/stable/overview.html#floating-objects)에 따릅니다. body 원점 속도와 CoM 속도를 혼동하지 않도록, 표면 속도는 body 원점의 선속도에 `ω × (표면 위치 − body 원점)`을 더합니다. CoM이 body 원점에서 떨어져 있어도 이 식은 유효합니다.

`state_delta(s, next_s)`의 출력은 다음 순서의 `12+2J`차원입니다.

1. 현재 root frame에서 본 위치 변화 3개
2. 상대 회전 `q_current⁻¹ q_next`의 회전 벡터 3개
3. 현재 root frame에서 본 world 선속도 변화 3개
4. 다음 시점의 각속도를 현재 root frame으로 옮긴 뒤 계산한 변화 3개
5. 관절 위치 변화 `J`개와 관절 속도 변화 `J`개

`apply_delta`는 같은 규약을 역으로 적용합니다. quaternion을 단순 덧셈하지 않고 상대 회전을 곱한 후 정규화합니다. `q`와 `−q`는 같은 학습 목표가 되며, 위치 변화를 `v*dt`에 강제로 묶지 않습니다. 마지막 성분까지 round-trip과 0 회전에서의 유한 gradient를 검사합니다.

## 형상에서 표면 포인트 만들기

MuJoCo가 컴파일한 body/joint/geom 메타데이터를 받아 PyTorch로 forward kinematics를 계산합니다. body의 기준 위치·회전, hinge pivot·axis·reference angle, 부모 연결을 모두 적용합니다. 모델 호출마다 MuJoCo의 동역학 solver를 실행하는 방식이 아닙니다.

| 형상 | 샘플링 단위 | 구현 이유 |
|---|---|---|
| Cube·직육면체 | 실제 여섯 면 | 임의의 mesh triangle 번호 대신 물리적 면 소속 보존 |
| sphere·ellipsoid | 정규화한 cube 표면 기반 여섯 patch | 일정한 수의 곡면 집약 단위 제공 |
| capsule·cylinder | 원주 네 patch와 양 끝 두 patch | Ant의 링크 표면을 같은 encoder 입력으로 변환 |
| 무한 plane | 현재 root를 투영한 위치 주변의 유한 격자 | 이동하는 물체 주변의 바닥 표현 유지 |
| 정적 box | 각 면의 격자 | 지지대·벽을 동일한 물리 포인트로 표현 |

큰 정적 box는 면의 법선 방향 위치를 유지하면서 접선 방향 격자를 현재 root 주변으로 이동합니다. 샘플은 항상 실제 box 범위 안에 있도록 제한합니다. 넓은 벽의 먼 모서리에만 포인트가 생기는 문제를 줄이며, 작은 지지대는 전체 면 격자가 유지됩니다.

면당 샘플 수 설정은 `ceil(sqrt(points_per_face))²`로 올림합니다. 격자 양 끝을 포함하므로 모서리와 꼭짓점도 샘플링합니다. 모서리의 같은 위치에 여러 면의 점이 존재할 수 있고, 각 점은 그 면의 법선과 소속을 유지합니다. 포인트마다 물체 질량을 나누어 배분하지 않습니다.

## 입력 성분

모든 포인트는 **현재 root body frame**을 공통 좌표계로 사용합니다. 정적 환경도 같은 좌표계에서 동적 표면과 함께 정렬합니다. 포인트마다 21개 실수 성분을 제공합니다.

| 성분 | 개수 | 의미 |
|---|---:|---|
| 위치 | 3 | 현재 root 기준 상대 위치, 실제 길이 단위 유지 |
| 법선 | 3 | 공통 frame으로 회전한 표면 법선 |
| 표면 속도 | 3 | world에서의 실제 표면 속도를 공통 frame으로 표현 |
| friction | 3 | sliding·torsional·rolling 마찰 설정 |
| solref | 2 | 접촉 응답의 time constant·damping 설정 |
| solimp | 5 | MuJoCo의 접촉 impedance 설정 |
| margin | 1 | 기하 접촉 여유 설정 |
| 정적 여부 | 1 | 정적 환경 1, 동적 표면 0 |

정적 바닥의 표면 속도는 0입니다. root가 움직인다는 이유로 정적 표면에 `−root_velocity`를 넣지 않습니다. 공통 frame으로 표현하는 것과 상대속도를 만드는 것은 구분하며, local attention의 상대속도 bias가 두 포인트의 속도 차를 사용합니다.

재질 설정을 포함하는 이유는 같은 위치·속도에서도 마찰이나 접촉 damping이 다르면 다음 상태가 달라지기 때문입니다. 예를 들어 반발 시나리오와 일반 착지 시나리오의 damping 차이를 모델에 숨기지 않습니다. `solref`는 반발계수 그 자체가 아니며, simulator의 실제 접촉 설정을 표현하는 입력입니다.

링크마다 별도로 42차원 문맥을 제공합니다. root 기준 링크 위치·자세, 선속도·각속도, 질량, root frame으로 변환한 3×3 관성, CoM offset, 중력, 외력·토크, dt, 관절축·관절 상태·motor torque가 들어갑니다. 관성의 principal-axis frame도 반영합니다. 질량·관성은 포인트 개수와 독립적입니다.

## Hilbert 정렬과 국소 attention 범위

`hilbert_keys`는 3차원 정수 좌표에 Skilling transpose 방식의 Hilbert 인덱스를 계산합니다. Morton 정렬을 Hilbert라고 부르지 않습니다. 두 방식은 `serialization` 설정으로 분리하며, 전체 4×4×4 정수 격자의 Hilbert 순서에서 연속한 점들이 항상 격자 이웃인지 검사합니다.

동적·정적 포인트 전체를 하나의 등방성 격자로 양자화하고 정렬합니다. layer별로 window 시작을 반 window만큼 이동합니다. 양 끝을 서로 이웃처럼 연결하는 cyclic wraparound는 쓰지 않습니다. point feature에는 상대 위치·속도·법선 관계를 사용한 학습 가능한 attention bias가 적용됩니다.

`model.coverage(state, action, threshold)`는 가까운 동적–정적 포인트 쌍 중 적어도 한 layer에서 같은 window에 포함된 비율을 반환합니다. 보고 항목은 `near_pairs`, `covered_pairs`, `coverage`입니다. 가까운 쌍이 아예 없으면 비율은 `NaN`이며, 이를 100% 성공으로 해석하지 않습니다. 이 지표는 직접 attention 범위이며 여러 layer를 거친 간접 정보 전달률과 다릅니다.

## 검증과 해석의 한계

Cube·두 링크·Ant에서 PyTorch FK 위치·회전을 MuJoCo의 `xpos/xquat`와 비교하고, 선속도·각속도를 MuJoCo body Jacobian에 `qvel`을 곱한 결과와 비교합니다. 다단계 rollout의 state gradient와 parameter gradient가 유한한지도 확인합니다. 큰 벽 샘플이 실제 box 면을 벗어나지 않는지 검사합니다.

양자화와 정렬의 이웃 선택 자체는 미분 가능하지 않습니다. 선택된 포인트 값, 위치, 회전, FK, 상태 갱신은 gradient를 유지하는 piecewise differentiable 경로입니다. 충돌을 분석적으로 풀거나 비관통을 보장하지 않습니다.

낮은 밀도의 smoke 격자는 실제 접촉 근처에 샘플이 없을 수 있습니다. 이때 낮은 근접쌍 수 또는 coverage를 숨기지 않고 기록해야 합니다. 제안서의 본 실험 설정인 면당 64점·바닥 256점과 작은 실행 점검 설정을 구분해야 합니다. mesh·heightfield 샘플링, 여러 free root, ball/slide joint는 현재 모델 경로의 지원 범위 밖이며 입력 시 오류를 냅니다.
