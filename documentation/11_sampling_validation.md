# 표면 샘플링과 독립 합산 검증

## 무엇을 구현하고 왜 측정했는가

`scripts/inspect_coverage.py`는 실제 가까운 동적 표면–환경 포인트 쌍이 local attention window에 함께 들어가는지를 측정합니다. 상위 aggregation이 입력 단계에서 누락된 모든 접촉 정보를 복구할 수 있다고 가정하지 않기 위해서입니다.

Cube 정지 장면의 4개 seed에서 거리 0.08m 이하의 sampled pair를 검사했습니다. 비율은 여러 seed의 분자를 더한 뒤 분모의 합으로 나눈 값입니다. 1층은 unshifted, 3층은 unshifted와 반-window shifted 분할의 합집합입니다. 3층의 마지막 분할은 첫 층과 같으므로 direct-pair coverage는 2층과 같습니다.

| sampling profile | 정렬 | local 층 | 근접쌍 수 | 직접 attention coverage |
|---|---|---:|---:|---:|
| physical_density | hilbert | 1 | 108 | 95.4% |
| physical_density | hilbert | 3 | 108 | 100.0% |
| physical_density | morton | 1 | 108 | 92.6% |
| physical_density | morton | 3 | 108 | 100.0% |
| smoke_density | hilbert | 1 | 48 | 68.8% |
| smoke_density | hilbert | 3 | 48 | 100.0% |
| smoke_density | morton | 1 | 48 | 66.7% |
| smoke_density | morton | 3 | 48 | 91.7% |
| undersampled_density | hilbert | 1 | 0 | 정의되지 않음 (근접쌍 0) |
| undersampled_density | hilbert | 3 | 0 | 정의되지 않음 (근접쌍 0) |
| undersampled_density | morton | 1 | 0 | 정의되지 않음 (근접쌍 0) |
| undersampled_density | morton | 3 | 0 | 정의되지 않음 (근접쌍 0) |
| undersampled_window64 | hilbert | 1 | 0 | 정의되지 않음 (근접쌍 0) |
| undersampled_window64 | hilbert | 3 | 0 | 정의되지 않음 (근접쌍 0) |
| undersampled_window64 | morton | 1 | 0 | 정의되지 않음 (근접쌍 0) |
| undersampled_window64 | morton | 3 | 0 | 정의되지 않음 (근접쌍 0) |

## 측정 결과를 반영한 설정

- `physical_density`: 면당 64·바닥 256·window 64.
- `smoke_density`: 면당 16·바닥 64·window 32. 최종 소규모 실험은 shifted local attention 2층을 사용합니다.
- `undersampled_density`: 면당 4·바닥 16. 실제 물리 접촉이 있어도 0.08m 이내의 sampled pair가 없습니다. window만 늘려도 샘플 자체의 누락을 해결할 수 없습니다.

이 검사 결과를 반영해 최종 smoke 설정의 표면 밀도를 높이고 모든 소규모 학습 결과를 다시 측정했습니다. 이는 무작정 모델 용량을 늘리기 전에 입력 접촉 표현을 검사해야 하는 이유입니다. 여기서 coverage 100%는 검사한 4개 정지 상태에만 해당하며, 모든 운동·형상에 대한 보장이 아닙니다.

## 독립 합산 비교군의 음성 대조

`independent`는 각 static geometry의 반응을 독립적으로 계산한 뒤 더합니다. 동일 상태에서 A/B/AB/∅를 바꾸는 8가지 검사에서 예측 world-twist 결합 norm의 최댓값은 **7.09e-08**, 허용오차는 2e-06였고 **통과=True**입니다.

이 검사는 학습 전 초기화된 모델의 구조적 합산 성질을 검증합니다. 예측 정확성이 좋다는 뜻이 아니며, 결합을 표현할 수 없는 비교군이 의도대로 제한됐는지 확인합니다. 선속도·각속도를 합친 norm은 수치적 0 검사에만 사용합니다.

```bash
docker compose run --rm research python scripts/inspect_coverage.py --output outputs/quick
docker compose run --rm research python -m cube report
docker compose run --rm research python -m cube docs
```

[Coverage 원시 JSON](results/coverage.json) · [독립 합산 원시 JSON](results/independent_null.json)
