# 형상·관절 구조 전이 학습

## 구현한 구성 요소

`cube.experiments.protocols.transfer`는 동일한 source checkpoint와 target dataset에서 다음 네 가지 적응 방법을 실행한다.

| 방법 | 초기화 | 학습하는 부분 |
|---|---|---|
| `scratch` | target 모델 무작위 초기화 | 전체 |
| `full` | shape이 같은 source parameter 복사 | 전체 |
| `point-frozen` | 같은 source parameter 복사 | Point encoder를 제외한 상위 aggregation/decoder |
| `aggregation-frozen` | 같은 source parameter 복사 | Point encoder만 |

Point encoder의 입력과 hidden 차원은 morphology와 독립적이다. 표면 patch/link/articulation 처리도 공유되는 구조를 사용한다. source와 target의 parameter 이름 및 shape을 확인해 호환되는 가중치만 복사하고, 복사/제외 목록과 trainable parameter 수를 기록한다. static geometry buffer는 학습 가중치로 복사하지 않는다. target 형상·질량·관성·관절 연결·환경 정보는 target simulator의 metadata로 재구성한다.

Cube/cuboid, 두 링크, Ant는 공통 state/action API를 사용하지만 generalized coordinate 차원이 다르다. 모델은 target scene에서 다시 생성되며 해당 차원의 decoder를 구성한다. shape이 달라지는 decoder가 있다면 무작위 초기화되고 transfer 보고서에 제외 이유가 나타난다. 이름이 같다는 이유로 잘못된 shape을 강제 복사하지 않는다.

**관절이 없던 Cube에서 관절체로 전이하는 예외:** source의 joint decoder는 한 번도 학습되지 않았으므로 shape이 같아도 복사하지 않는다. target의 새 joint output head는 모든 적응 방법에서 학습 가능하게 둔다. 따라서 이 경우 `aggregation-frozen`은 **사전학습된 aggregation 고정 + Point encoder와 새 joint head 학습**을 뜻한다. 이 예외를 checkpoint 및 결과의 `novel_head_policy`에 기록한다. 기존 관절체 사이의 동일 구조 전이에서는 지정한 freeze 정책을 그대로 따른다.

각 데이터 예산과 seed에서 동일한 training trajectory ID를 선택한다. validation/test는 이 subset 밖의 독립 trajectory다. 정규화 통계도 target training subset에만 다시 적합하며 source 또는 target test 통계를 섞지 않는다. 전체 source architecture를 scratch 기준선에도 사용하므로 source 초기화와 freeze 정책의 영향을 비교할 수 있다.

## 왜 구현했는가

연구 제안에서 재사용하려는 것은 독립적인 point 접촉 충격량 해답이 아니라 국소 접촉 상황의 표현이다. 따라서 단순히 전체 model을 fine-tuning하는 결과만으로는 Point 표현과 aggregation의 역할을 구분할 수 없다. 상위 부분만 적응시키는 방법과 Point 부분만 적응시키는 방법을 같은 데이터에서 비교해야 구조 분리의 효용을 해석할 수 있다.

## 실험 구성

`cuboid_transfer.json`은 큐브에서 종횡비와 질량이 다른 직육면체로 전이하기 위한 target data다. `two_link_smoke/research.json`과 `ant_smoke/research.json`은 연결 구조가 달라지는 target data다. 각 target dataset을 먼저 생성한 뒤 같은 source checkpoint로 transfer 명령을 실행한다. Cube → 두 링크에서 선택된 checkpoint를 Ant의 source로 사용하면 순차 전이도 가능하다.

`transfer_smoke.json`은 2/4개 trajectory, seed 1개, epoch 2개의 실행 검증용 설정이다. `transfer_research.json`은 4/16/64개 trajectory와 3개 seed를 비교한다. 요청한 예산보다 train trajectory가 적으면 오류를 내므로 충분한 target data를 먼저 생성해야 한다.

마찰/재질을 source와 같게 유지하는 구조 전이와, target data의 scene friction을 바꾸는 물성 전이는 별도 output에 저장해야 한다. 결과 파일이 다른 설정을 자동으로 같은 실험으로 합치지 않는다. Point를 고정한 방법이 모든 재질 변화에 유리하다고 가정하지 않는다.

## 결과 해석과 한계

각 run은 checkpoint와 evaluation을 남기고 `transfer.json`은 모델·예산별 평균/표준편차 및 parameter 복사 보고서를 집계한다. 실행되지 않은 grid와 실행 중인 grid는 성능 결과가 아니며 `completed` 필드로 구분한다. 공식 NeRD 등의 사전학습 가중치로부터 전이를 구현했다고 주장하지 않는다. 현 구현은 이 저장소의 학습된 dynamics checkpoint 간 전이다.

few-shot 성능 향상, 새로운 morphology에서의 안정적 장기 rollout, MPC 제어 개선은 실제 결과로 검증해야 하는 가설이다. 작은 smoke 실행은 데이터·학습·전이·평가 경로가 연결되어 있다는 확인이며 논문 수준의 샘플 효율 결론은 아니다.
