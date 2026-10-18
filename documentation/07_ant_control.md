# Ant 사족 보행과 학습 모델 기반 제어

## 구현한 구성 요소

Ant는 하나의 자유 base와 8개 hinge joint를 가진 관절체다. 각 다리의 hip/knee가 순서대로 배치되며 입력은 `[-1,1]`의 8개 motor control이다. MJCF의 기본 gear 20을 통해 토크로 변환된다. 출력 상태는 base pose/velocity 및 일반화 관절 상태이며 각 link의 독립 pose를 예측하지 않는다.

`ant_gait_action`은 대각선 다리쌍의 위상을 맞춘 trot reference와 PD feedback을 구현한다. 좌우 hip 부호를 다르게 적용해 발의 stance sweep이 같은 진행 방향을 만들고, swing 시 knee를 접는다. 기본 frequency 1.4 Hz, hip amplitude 0.35 rad, knee lift 0.4 rad, PD gain 12/0.8 및 gear 20을 사용한다. 이는 **수작업으로 설계한 실행 가능한 보행 기준선**이다. 학습된 policy라고 표기하지 않는다.

Ant 데이터는 정상 gait, gait에 탐색 action을 더한 경우, 무작위 motor action의 세 종류를 번갈아 수집한다. `walk/rest/free_fall` 환경과 서로 다른 초기 seed를 이용할 수 있다. 정규화 control을 저장하므로 model action과 실제 simulator action의 단위가 일치한다.

`shooting_mpc`는 학습한 world model을 사용한 CEM 방식의 action sequence 탐색이다. 매 control step마다 후보 sequence를 생성하고 동일한 horizon에서 모델 rollout의 비용을 비교한다. 우수한 후보의 평균과 표준편차로 다음 탐색 분포를 업데이트한다. 최선 sequence의 첫 action만 독립적인 MuJoCo 또는 Warp simulator에 적용한다.

MPC rollout의 매 단계는 **예측 상태에서 표면 point, joint FK와 표면 속도를 다시 계산**한다. 모델이 현재 상태로 제공받는 contact feature는 현재 geometry에서 계산하며 simulator의 미래 contact/impulse는 넣지 않는다. 과거 action history와 각 후보 sequence의 action history도 일관되게 유지한다.

Cube는 목표 위치와 속도/행동 비용을 이용한다. Ant는 전방 속도 보상, trunk 높이·기울기, lateral deviation, action 비용을 사용한다. Ant의 MPC 탐색은 gait를 기본 action proposal로 사용하므로, 학습 모델이 짧은 horizon에서 검토할 수 있는 보행 근처의 action을 탐색한다. 학습 초기의 부정확한 모델은 baseline보다 나쁜 action을 선택할 수 있다.

Cube baseline은 위치 gain 40, 속도 damping 8 및 힘 한계 20 N을 사용하여 기본 질량·마찰의 정지 마찰력을 넘을 수 있게 했다. MPC도 힘 20 N, 토크 2 Nm 범위를 사용한다. smoke 데이터의 작은 random wrench 범위보다 큰 제어 입력이므로 실제 제어 연구에서는 해당 force/action 분포의 학습 데이터를 별도로 충분히 생성해야 한다.

## 왜 구현했는가

단일 강체의 접촉 표현이 관절체의 generalized state 예측으로 연결되는지 확인하기 위해 두 링크 다음에 Ant를 구성했다. 보행 baseline은 MJCF가 실제로 이동 가능한지와 모델 학습 데이터를 만들 수 있는지 검증한다. 모델 MPC는 예측 정확도 개선이 외부 물리 환경의 제어 성능으로 이어지는지 검증한다. 모델 안에서 계산한 보상만으로 성공을 판정하지 않는다.

## 평가와 산출물

`control`은 `baseline`, `zero`, `mpc`를 지원하며 실제 solver에서 전체 rollout을 실행한다. `rollout.npz`에 states/actions, `scene.json`에 scene 설정, `control.json`에 다음 지표를 저장한다.

- ground-truth return, 전방 이동 거리와 평균 속도
- Ant의 낮은 trunk 높이 또는 기울기에 따른 fall fraction
- Cube의 목표 위치 거리, Ant의 전방 0.15 m 이상 이동과 무전도 여부로 정의한 success
- 최대 관통, 평균 action 제곱, control planning wall time
- MPC horizon/candidates/iterations와 매 step 모델 transition 계산량

MPC 모델을 비교할 때 같은 seed, scene, step 수, horizon, 후보 수, CEM iteration을 사용해야 한다. baseline과 MPC의 계산량이 같다고 주장하지 않고 비용을 같이 보고한다. 필요하면 저장한 trajectory를 root의 headless rendering 명령으로 MP4로 만들거나 `video=True`의 GIF 경로를 사용할 수 있다.

`configs/ant_smoke.json`은 작은 실행 검증용 데이터다. `ant_research.json`은 더 많은 보행/접촉 전환 데이터를 생성한다. source Cube checkpoint의 전이와 Ant scratch 학습을 모두 지원하지만, 짧은 smoke 학습 결과를 안정적인 보행 정책 학습 완료로 간주하면 안 된다. motor gear를 변경했다면 gait의 토크 정규화와 gain도 맞추어야 한다.
