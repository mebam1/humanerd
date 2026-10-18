"""Publish measured CUDA checks separately from the full, unfinished research suite."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cube.documentation import build_docs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    suite = read("outputs/gpu-check/suite.json")
    if not suite["completed"] or suite["completed_runs"] != 7:
        raise ValueError("Finish the seven-model GPU check before building its measured report")
    capacity = read("outputs/runtime/gpu_capacity_b32.json")
    resume = read("outputs/gpu-resume-check/verification.json")
    data = read("data/cube-nerd/manifest.json")
    check_config = read("configs/suite_gpu_check.json")
    tests = ET.parse("outputs/runtime/pytest.xml").getroot()
    test_suite = tests.find("testsuite") if tests.tag != "testsuite" else tests
    assert int(test_suite.attrib["failures"]) == int(test_suite.attrib["errors"]) == 0
    training = {row["model"]: read(Path(row["checkpoint"]).parent / "training.json") for row in suite["runs"]}
    output = Path("docs/assets")
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for axis, (name, result) in zip(axes.flat, training.items()):
        epochs = result["epochs"]
        axis.plot([r["epoch"] for r in epochs], [r["train_loss"] for r in epochs], "o-", label="Train")
        axis.plot([r["epoch"] for r in epochs], [r["val_loss"] for r in epochs], "s-", label="Validation")
        axis.set(title=name, xlabel="Epoch", ylabel="Normalized loss", xticks=[1, 2, 3], yscale="log")
        axis.grid(alpha=.2)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False)
    figure.suptitle("CUDA runtime check: 32 train trajectories, 3 epochs, 1 seed (not a converged study)")
    figure.savefig(output / "gpu-check-learning.png")
    plt.close(figure)
    names = list(training)
    positions = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, section, title in zip(axes, ["one_step", "rollout"], ["One-step position RMSE", "16-step rollout position RMSE"]):
        for offset, split in [(-.2, "test"), (.2, "ood")]:
            values = [100*row["evaluation_metrics"][f"{split}/{section}/position_rmse"] for row in suite["runs"]]
            axis.bar(positions+offset, values, .4, label=split.upper())
        axis.set(title=title, ylabel="cm (lower is better)", xticks=positions, xticklabels=names)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=.2)
        axis.legend(frameon=False)
    figure.suptitle("8 held-out trajectories per split; one seed; no superiority claim")
    figure.savefig(output / "gpu-check-errors.png")
    plt.close(figure)

    accuracy_rows, train_rows = [], []
    for row in suite["runs"]:
        values = row["evaluation_metrics"]
        accuracy_rows.append(f"| {row['model']} | {100*values['test/one_step/position_rmse']:.3f} | "
                            f"{100*values['ood/one_step/position_rmse']:.3f} | "
                            f"{100*values['test/rollout/position_rmse']:.2f} | {100*values['ood/rollout/position_rmse']:.2f} |")
        record = training[row["model"]]
        train_rows.append(f"| {row['model']} | {record['epochs'][0]['val_loss']:.4f} → {record['epochs'][-1]['val_loss']:.4f} | "
                          f"{record['best_val_loss']:.4f} | {record['elapsed_seconds']:.1f} | {row['parameters']:,} |")
    resumed_message = f"{resume['resumed_max_weight_difference']:g}"
    smallest_id = min(suite["runs"], key=lambda row: row["metrics"]["position_rmse"])
    hierarchy = next(row for row in suite["runs"] if row["model"] == "hierarchical")
    text = f"""# 실제 GPU 실행과 대규모 데이터 준비 결과

이번 기록은 **RTX 4060의 실제 CUDA 학습 경로 검증**이다. 본 실험용 데이터는 모두 생성했지만, **10K trajectories × 50 epochs × 21회 비교는 아직 실행하지 않았다.** 아래 짧은 학습은 모델 우열이나 수렴을 입증하지 않는다. 장기 실행 절차는 [GPU 학습 안내](12_gpu_training.md)를 따른다.

## 실제로 완료한 작업

| 항목 | 측정 결과 |
|---|---|
| 데이터 | train 10,000 + validation/test/OOD 각 1,000 = 13,000 trajectories |
| 길이 | 각 100 transitions, 101 states; 학습용 1,000,000 transitions |
| 생성 시간 | {data['elapsed_seconds']:.2f}초, MuJoCo worker 8개 |
| 저장 | 51 NPY shards, float32, 배열 payload 128,232,000 bytes; JSON/header 별도 |
| GPU | {capacity['gpu']}, {capacity['gpu_total_memory_mb']:.1f} MiB |
| 학습 환경 | PyTorch {capacity['torch_version']}, bf16 |
| 자동 검증 | {test_suite.attrib['tests']} tests, failures/errors 0; 실제 Warp CUDA 포함 |
| CUDA 학습 재개 | 1 update 후 재개한 2-update 결과와 연속 학습의 최대 가중치 차이 **{resumed_message}** |
| W&B | 실제 SDK offline 기록·run ID 재사용 확인; online 연결은 인증 미설정으로 미검증 |

완료된 데이터는 `data/cube-nerd/`에 있다. 같은 설정으로 `generate --resume`를 실행하면 완료 파일을 검증하고 재사용하며 manifest의 학습 식별자를 바꾸지 않는다. raw 결과는 `outputs/gpu-check/`, `outputs/gpu-resume-check/`, `outputs/runtime/`에 보존했다. [측정 JSON 다운로드](results/gpu_check.json)에도 실험 설정과 지표를 함께 저장했다.

## 짧은 비교의 통제 조건

7개 모델은 동일한 training trajectory 32개, seed 0, 3 epochs, full-resolution point 입력과 width 128을 사용했다. 각 epoch는 3,200 transitions를 학습했고, batch 32 × accumulation 4로 nominal batch 128을 구성했다. scene별 배치들을 섞으며 마지막 작은 배치는 실제 표본 수로 나눈다. validation/test는 각 8개의 고정된 시나리오별 subset, OOD는 floor_wall 8개다. 평가는 validation으로 선택한 `best.pt`를 사용한다. rollout 길이는 **16 steps**이며 장기 본 실험의 100-step 평가와 다르다.

본 설정은 [suite_gpu_check.json](../configs/suite_gpu_check.json)의 실행 검증용이다. 본 비교의 10,000 trajectories, 50 epochs, nominal batch 512, 3 seeds보다 훨씬 작다. 평균화한 수치는 trajectory별 metric의 산술 평균이며 1 seed에서 통계적 유의성을 계산하지 않는다.

![짧은 CUDA 학습 곡선](assets/gpu-check-learning.png)

| 모델 | Epoch 1 → 3 val loss | Best val loss | 학습 기록 시간(s) | Parameters |
|---|---:|---:|---:|---:|
{chr(10).join(train_rows)}

시간은 `training.json.elapsed_seconds`로 정규화·학습·validation·checkpoint 저장을 포함한다. 모델 구성, W&B 초기화/종료와 최종 test/OOD 평가는 별도다. 긴 학습의 정확한 소요 시간 예측치로 쓰지 않는다.

## One-step과 자율 rollout

![실제 test와 OOD 오차](assets/gpu-check-errors.png)

| 모델 | ID one-step 위치(cm) | OOD one-step 위치(cm) | ID rollout 위치(cm) | OOD rollout 위치(cm) |
|---|---:|---:|---:|---:|
{chr(10).join(accuracy_rows)}

이번 짧은 실행에서 ID one-step 위치 오차가 가장 작은 모델은 `{smallest_id['model']}`({100*smallest_id['metrics']['position_rmse']:.2f} cm)이며, 계층형은 {100*hierarchy['metrics']['position_rmse']:.2f} cm였다. **ID one-step에서는 계층형의 우위가 관찰되지 않았다.** 계층형의 ID 16-step rollout 위치 오차는 {100*hierarchy['evaluation_metrics']['test/rollout/position_rmse']:.2f} cm로 증가했다. OOD 16-step rollout에서는 계층형이 {100*hierarchy['evaluation_metrics']['ood/rollout/position_rmse']:.2f} cm로 이 표의 다른 모델보다 작았다. 지표에 따라 순위가 달라지므로 하나의 평균 오차로 전체 성능을 판단할 수 없다. 구조의 우열을 결론낼 학습량·seed 반복은 아직 부족하다.

One-step 오차가 작아도 예측 상태를 다시 입력하는 rollout에서는 오차가 누적된다. 이 표의 OOD가 일부 ID보다 쉬워 보일 수 있지만, 두 split은 서로 다른 시나리오 분포다. OOD 값이 작다는 사실만으로 조합 일반화가 해결됐다고 해석하지 않는다. 학습량·seed·평가 trajectory가 부족하므로 이 순위를 계층형 구조의 최종 성능 순위로 사용하지 않는다. 본 실험에서는 validation 수렴, 3 seeds의 변동, 100-step rollout, 접촉 진입/유지 구간, A/B 결합 개입을 함께 평가한다.

## 재현 명령과 다음 단계

```bash
docker compose --profile cuda run --rm cuda python -m cube suite --config configs/suite_gpu_check.json --output outputs/gpu-check --resume
docker compose --profile cuda run --rm cuda python scripts/report_gpu_check.py
docker compose up -d docs
```

동일 경로에서 완료된 suite는 학습을 반복하지 않는다. GPU check를 새로 측정하려면 다른 출력 폴더를 사용하고 report 스크립트의 입력 경로도 조정한다. 본 비교는 `.env`에 W&B 인증을 설정한 뒤 실행한다.

```bash
docker compose --profile cuda run --rm cuda python -m cube suite --config configs/suite_gpu.json --output outputs/cube-gpu-suite --resume
```

먼저 계층형만 진행하려면 `--models hierarchical`을 추가한다. 기존 [소규모 CPU 실험](10_results.md)은 당시의 설정·결과로 보존하며, 이 GPU 기록과 동일한 실험인 것처럼 합산하지 않는다.
"""
    # Docs server serves docs/, so source configs are plain references rather than broken links.
    text = text.replace("[suite_gpu_check.json](../configs/suite_gpu_check.json)", "`configs/suite_gpu_check.json`")
    Path("documentation/13_gpu_results.md").write_text(text, encoding="utf-8")
    Path("docs/results/gpu_check.json").write_text(json.dumps({"data": {key: data[key] for key in
           ("splits", "expected_trajectories", "elapsed_seconds", "dataset_fingerprint")},
           "suite": suite, "capacity": capacity, "resume": resume, "check_config": check_config}, indent=2), encoding="utf-8")
    print(build_docs())


if __name__ == "__main__":
    main()
