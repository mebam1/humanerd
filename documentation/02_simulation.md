# 공통 물리 시뮬레이터와 장면

## 무엇을 구현했는가

`src/cube/sim/scenes.py`는 `SceneSpec`에서 MJCF를 생성한다. 같은 모델을 MuJoCo의 C 엔진 또는 `mujoco_warp`의 Warp 커널로 실행한다. `Simulator`는 두 엔진에 동일한 상태·행동·reset 인터페이스를 제공한다. Warp 요청 시 Warp 커널로 실제 적분하며, 사용할 수 없는 CUDA 장치나 패키지는 오류로 알린다.

| 논리적 구성 요소 | 구현 | 구현 이유 |
|---|---|---|
| 직렬화 가능한 실험 명세 | 종류, 초기화 seed, 크기, 질량, 마찰, 중력, 접촉 파라미터, dt와 substeps | 저장된 trajectory의 물리 조건을 재구성한다. |
| 일관된 상태 경계 | `qpos` 뒤에 `qvel`을 연결, 루트 free joint 뒤에 hinge | 상태 기반 모델과 두 물리 엔진 사이의 좌표계 혼동을 방지한다. |
| 접촉 시나리오 | 자유 낙하, 최초 충돌, 반발, 정착, 정지, sliding, 모서리 회전, 바닥+벽, 벽만 | 접촉 종류와 전환별 성능을 따로 검증한다. |
| 물리적 개입 | 지지 없음/A/B/AB, 같은 면/다른 면 | 동일한 현재 상태에서 원격 지지 추가에 따른 비가산 반응을 측정한다. |
| 구조 확장 | 직육면체, 1 hinge 두 링크, 8 hinge 사족 Ant | 국소 표현을 유지하며 연결 구조가 변할 때 전이를 시험한다. |
| 기구학 메타데이터 | body의 질량·관성·부모, joint의 축·주소, geom의 형상·물성 | 예측 상태에서 표면을 다시 만들고 링크·관절 정보를 모델에 전달한다. |
| 현재 접촉 진단 | 거리·법선·위치·현재 법선 상대속도 | 관통과 접촉 사건을 평가하되 solver 충격량을 학습 입력에 넣지 않는다. |

## 상태와 시간의 정의

단일 강체 상태는 13차원이다. `qpos = [x,y,z,qw,qx,qy,qz]`, `qvel = [vx,vy,vz,wx,wy,wz]`이며 관절체는 각 배열 끝에 관절 위치/속도가 붙는다. 선속도는 world frame, 루트 각속도는 body frame인 MuJoCo free joint의 원래 규약을 따른다. 루트 자세는 body frame을 world frame으로 변환한다. 루트 위치와 질량중심이 다른 링크의 표면 속도는 기구학에서 계산하며 루트 속도를 모든 링크의 COM 속도로 간주하지 않는다.

`dt`는 데이터와 제어의 시간 간격이고 물리 적분 간격은 `dt/substeps`이다. 기본값은 0.02초/4회이다. Euler 적분, Newton 접촉 solver, pyramidal friction을 두 backend에 동일하게 지정한다. Warmstart를 끄고 reset에서 내부 상태를 초기화하여 동일한 현재 상태의 다음 상태를 재현한다. MuJoCo와 Warp는 정밀도와 solver 구현 차이 때문에 bitwise 일치를 요구하지 않는다.

정지 Cube에도 작은 초기 평면 위치·yaw·수평속도 차이를 주며, Ant 보행 초기화에는 작은 위치·yaw·관절각 차이를 준다. seed가 다른 episode가 완전히 같은 궤적이 되어 train/test에 복제되는 것을 방지한다. 개입 실험에서는 같은 seed를 네 조건에 공유하므로 이 구분이 반사실 상태 일치를 훼손하지 않는다.

Cube 행동은 world frame의 힘 3축과 COM에 작용하는 토크 3축이다. 관절체 행동은 `[-1,1]`의 motor 입력이며 실제 토크는 gear(기본 20)를 곱한 값이다. 이 단위는 데이터 생성, 모델 문맥, 제어에서 공유한다.

## 접촉과 개입의 세부사항

MuJoCo의 `solref`는 반발계수가 아니다. 기본 contact damping ratio는 1이고, bounce/rebound 장면에서는 0.25로 낮춰 반발을 만든다. contact time constant와 damping, margin을 명세에 저장하므로 재질을 바꾼 실험에서 이 설정도 함께 기록한다.

`contact_margin`은 각 geom의 margin이다. 실제 접촉 쌍의 포함 거리는 MuJoCo가 조합하며 기본 두 geom에서는 0.002m가 된다. 따라서 진단의 접촉 수는 단일 geom margin이 아닌 접촉 쌍의 `includemargin`으로 판정한다. 관통 깊이는 margin을 빼지 않은 물리 표면 거리의 음수 부분이다.

같은 면 개입은 큐브 바닥의 좌우에 분리된 두 지지대를 둔다. 다른 면 개입은 A를 아래쪽, B를 오른쪽 면에 둔다. 모든 조건은 같은 위치·속도·질량·형상·A 또는 B 지지 형상을 유지하며, 존재 여부만 달라진다. 바닥은 제거되어 숨은 지지 경로가 생기지 않는다. 반사실 실험에 필요한 상태 일치와 지지대 동일성을 테스트한다.

`contact_info()`는 현재 상태의 기하 접촉과 Jacobian으로 계산한 상대속도를 반환한다. Warp 실행 후 host에서 계산하는 값은 진단용 기구학/충돌 검사이며, host MuJoCo의 적분이나 contact force로 Warp 결과를 대체하지 않는다. 입력 표면은 이 진단의 impulse/force가 아니라 기하 메타데이터와 현재 상태에서 구성한다.

`step(..., record_substeps=True)`는 시작과 각 물리 substep 종료의 접촉·접근속도를 `last_substep_contacts`에 보관한다. 제어 step의 양 끝에서는 보이지 않는 충돌 후 분리를 사건 label에서 누락하지 않기 위한 옵션이다. 데이터 생성용 label이며 모델 입력 경로와 분리한다. Warp에서는 이 옵션에 매 substep host 동기화 비용이 발생하므로, 순수 실행 throughput과 label 수집 포함 시간을 구분해야 한다. 해상도는 물리 timestep이며 연속시간 충돌 검출을 보장하지 않는다.

## Ant 구성

Ant는 외부 모델 파일을 복제하지 않고 작성한 간단한 box torso와 capsule 다리 네 개이다. 다리마다 z축 hip yaw와 수평 접선축 knee hinge가 있다. 순서는 front-left hip/knee, front-right hip/knee, back-left hip/knee, back-right hip/knee이고 총 상태는 `nq=15`, `nv=14`이다. 기본 관절 각도는 모두 0이며, 이때 torso 높이는 0.34m이다. 총 질량은 `SceneSpec.mass`이고 torso에 절반, 8개 다리에 나머지를 배분한다.

이는 연구용 사족 형태와 토크 제어 과제이며 특정 실물 Ant 하드웨어의 동역학을 보정한 모델은 아니다. 보행 성능은 별도의 ground-truth rollout에서 거리·낙상·높이 등을 측정한다.

## 검증과 근거

물리 테스트는 자유 낙하의 해석적 속도·Euler 위치, world frame 힘, 접촉 transition replay, 정지 지지, sliding 감속, 실제 반발, 개입 비가산성, Ant 자유도와 질량을 확인한다. Warp 통합 테스트는 명시적 opt-in으로 Cube와 Ant의 실제 커널 실행과 비접촉 구간 수치 일치를 검사한다. 실행 명령은 루트 README의 Docker Compose 검증 절을 사용한다.

구현 API를 확인한 공식 자료: [MuJoCo Warp 공개 API](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/__init__.py), [MuJoCo Warp 모델·데이터 업로드](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/io.py), [MuJoCo 계산과 상태 규약](https://mujoco.readthedocs.io/en/stable/computation/index.html).
