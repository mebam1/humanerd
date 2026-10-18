"""Generate static scientific figures and a report strictly from measured JSON/NPZ."""
from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_report(output="outputs/quick"):
    root = Path(output)
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    def read(relative):
        return json.loads((root / relative).read_text(encoding="utf-8"))
    suite, coupling, transfer = read("suite/suite.json"), read("coupling_model/coupling.json"), read("transfer/transfer.json")
    ant, zero = read("ant_baseline/control.json"), read("ant_zero/control.json")
    short, mpc = read("ant_baseline_short/control.json"), read("ant_mpc/control.json")
    evaluation = read("ant_evaluation/evaluation.json")
    validation, metadata = read("validation/validation.json"), read("run_metadata.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    palette = ["#207b8d", "#d79a47", "#6c74bd", "#c56978"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for i, name in enumerate(["state_mlp", "deepsets", "hierarchical"]):
        rows = sorted([r for r in suite["summary"] if r["model"] == name], key=lambda r: r["budget"])
        for axis, metric, title in zip(axes, ["linear_velocity_rmse", "position_rmse"],
                                       ["One-step linear velocity RMSE (m/s)", "One-step position RMSE (m)"]):
            axis.errorbar([r["budget"] for r in rows], [r["metrics"][metric]["mean"] for r in rows],
                          yerr=[r["metrics"][metric]["std"] for r in rows], marker="o", capsize=4, label=name, color=palette[i])
            axis.set(title=title, xlabel="Training trajectories (24 steps each)", xticks=[4, 8])
            axis.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.savefig(assets / "cube-comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, layout in zip(axes, ["same_face", "cross_face"]):
        rows = [r for r in coupling["records"] if r["layout"] == layout]
        truth = np.array([r["ground_truth_C_AB"] for r in rows])
        prediction = np.array([r["predicted_C_AB"] for r in rows])
        x = np.arange(6)
        axis.bar(x-.18, truth.mean(0), .36, yerr=truth.std(0), color=palette[0], label="Simulator C_AB")
        axis.bar(x+.18, prediction.mean(0), .36, color=palette[1], label="3-epoch model C_AB")
        axis.set(title=layout.replace("_", " ").title(), xticks=x, xticklabels=["vx", "vy", "vz", "wx", "wy", "wz"],
                 ylabel="Linear: m/s; angular: rad/s")
        axis.axhline(0, color="#999", linewidth=.6)
        axis.grid(axis="y", alpha=.18)
    axes[0].legend(fontsize=8)
    fig.savefig(assets / "coupling.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for i, (folder, label) in enumerate([("ant_baseline", "PD trot (6s)"), ("ant_zero", "Zero torque (6s)"), ("ant_mpc", "Model MPC (1s)")]):
        states = np.load(root / folder / "rollout.npz")["states"]
        times = np.arange(len(states)) * .02
        axes[0].plot(times, states[:, 0]-states[0, 0], color=palette[i], label=label)
        axes[1].plot(times, states[:, 2], color=palette[i], label=label)
    for axis, title, ylabel in zip(axes, ["Ant forward displacement", "Ant torso height"], ["Displacement (m)", "Height (m)"]):
        axis.set(title=title, xlabel="Time (s)", ylabel=ylabel)
        axis.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.savefig(assets / "ant-control.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for i, mode in enumerate(["scratch", "full", "point-frozen", "aggregation-frozen"]):
        rows = sorted([r for r in transfer["summary"] if r["model"] == mode], key=lambda r: r["budget"])
        axis.plot([r["budget"] for r in rows], [r["metrics"]["linear_velocity_rmse"]["mean"] for r in rows],
                  "o-", label=mode, color=palette[i])
    axis.set(title="Cube → two-link adaptation (2 epochs, one seed)", xlabel="Target training trajectories",
             ylabel="Linear velocity RMSE (m/s)", xticks=[2, 4])
    axis.grid(alpha=.18)
    axis.legend(fontsize=8)
    fig.savefig(assets / "transfer.png", dpi=160)
    plt.close(fig)

    rows = []
    for r in suite["summary"]:
        m = r["metrics"]
        runs = [run for run in suite["runs"] if run["model"] == r["model"] and run["budget"] == r["budget"]]
        rows.append(f"| {r['model']} | {r['budget']} | {m['linear_velocity_rmse']['mean']:.4f} ± {m['linear_velocity_rmse']['std']:.4f} | {m['position_rmse']['mean']:.5f} | {np.mean([x['parameters'] for x in runs]):.0f} | {np.mean([x['latency_ms'] for x in runs]):.2f} |")
    test_rows = []
    for r in suite["runs"]:
        if r["budget"] == 8 and r["seed"] == 0:
            test_rows.append(f"| {r['model']} | {r['metrics']['linear_velocity_rmse']:.4f} | {r['ood_metrics']['linear_velocity_rmse']:.4f} |")
    transfer_rows = []
    for r in transfer["summary"]:
        transfer_rows.append(f"| {r['model']} | {r['budget']} | {r['metrics']['linear_velocity_rmse']['mean']:.4f} | {r['metrics']['joint_position_rmse']['mean']:.4f} |")
    warp = read("warp_validation/validation.json") if metadata.get("warp_validation_requested") and (root / "warp_validation/validation.json").exists() else None
    coupling_rows = []
    for layout in ["same_face", "cross_face"]:
        records = [r for r in coupling["records"] if r["layout"] == layout]
        truth = np.array([r["ground_truth_C_AB"] for r in records])
        error = np.array([r["predicted_C_AB"] for r in records])-truth
        coupling_rows.append(f"| {layout} | {np.linalg.norm(truth[:,:3],axis=1).mean():.4f} | {np.linalg.norm(truth[:,3:],axis=1).mean():.4f} | {np.linalg.norm(error[:,:3],axis=1).mean():.4f} | {np.linalg.norm(error[:,3:],axis=1).mean():.4f} |")
    cube_baseline, cube_mpc = read("cube_baseline/control.json"), read("cube_mpc/control.json")
    text = f'''# 직접 실행한 실험과 결과 분석

> 실행 시각: **{metadata['timestamp_kst']}**. 모든 수치는 저장된 JSON·NPZ에서 생성했습니다. smoke 규모의 실행 검증이며 논문 가설의 성공을 선언하는 결과가 아닙니다.

## 1. 실행 환경과 검증 범위

Docker Compose의 루트 마운트 환경에서 CPU 신경망 학습과 MuJoCo 실험을 수행했습니다. Warp 검증은 {'RTX 4060 CUDA에서 수행했습니다' if warp else '이 보고서 실행에 포함되지 않았습니다'}.

| 패키지 | 실제 설치 버전 |
|---|---|
''' + "\n".join(f"| {name} | {version} |" for name, version in metadata["packages"].items()) + f'''

검증 범위: Cube 세 모델 × 데이터 예산 2종 × seed 2개 = **12회 학습**, Cube→두 링크 4가지 적응 × 예산 2종 = **8회 학습**, Ant dynamics 학습·rollout·제어. 짧은 실험 전체 소요 시간은 {metadata['elapsed_seconds']:.1f}초이며, 컨테이너 빌드·최초 kernel compilation·렌더링 시간은 제외했습니다.

## 2. 물리 상태와 재현성

- MuJoCo Cube·두 링크·Ant 같은 상태 재실행: 최대 오차 **{max(r['repeat_transition_max_abs'] for r in validation['checks']):.3g}**.
- 표면 속도와 위치 유한차분 비교: 최대 오차 **{validation['point_velocity_finite_difference_error']:.3g} m/s**.
- Warp 상태 재실행 검사: **{str(warp['passed']) if warp else '미실행'}**. 이 작은 검사 통과가 GPU의 모든 실행에서 결정성을 보장한다는 뜻은 아닙니다.
- 상태 전환 0.02초, 내부 물리 step 0.005초. quaternion 순서 wxyz와 base angular velocity의 로컬 좌표계를 검사했습니다.

## 3. Cube 모델 비교

같은 trajectory 예산과 seed, optimizer, 3 epochs를 사용했습니다. 비교군마다 실제 파라미터 수와 latency를 기록하며 계산량이 같다고 가정하지 않습니다. 오차의 ±는 두 seed의 **표준편차**입니다. 데이터는 4개 시나리오, trajectory당 24 transitions이며 test는 독립 trajectory입니다.

| 모델 | train trajectories | 선속도 RMSE (m/s) | 위치 RMSE (m) | 파라미터 | CPU latency (ms) |
|---|---:|---:|---:|---:|---:|
''' + "\n".join(rows) + '''

![Cube sample-budget comparison](assets/cube-comparison.png)

**해석:** 이 표는 학습과 비교 파이프라인이 동작한다는 증거입니다. 3 epochs와 두 seed만으로 계층형 구조의 우월성이나 표본 효율을 판단할 수 없습니다. 모델 크기와 입력 요약 방식도 다르므로 단일 평균 수치로 결론을 내리지 않습니다. 면당 16포인트·바닥 64포인트·local attention 2층의 smoke 설정은 면당 64포인트의 본 실험과 다릅니다.

### 훈련에서 제외한 바닥–벽 동시 접촉

아래는 budget 8, seed 0입니다. floor contact와 wall-only는 훈련에 있지만 동시 조합은 OOD에만 포함했습니다. ID/OOD의 접촉 강도와 분포가 다르므로 OOD 오차가 반드시 더 커야 하는 것은 아닙니다.

| 모델 | ID 선속도 RMSE | OOD 선속도 RMSE |
|---|---:|---:|
''' + "\n".join(test_rows) + '''

## 4. 다중 접촉 개입: 독립 합산이 충분한가

각 seed에서 초기 상태·물성·action을 같게 고정하고 지지점만 ∅, A, B, A+B로 바꾸었습니다. `C_AB = Δν_AB − Δν_A − Δν_B + Δν_∅`입니다. 선속도와 각속도의 단위가 다르므로 norm을 따로 보고합니다.

| 배치 | 실제 선속도 결합 norm (m/s) | 실제 각속도 결합 norm (rad/s) | 모델 선속도 결합 오차 | 모델 각속도 결합 오차 |
|---|---:|---:|---:|---:|
''' + "\n".join(coupling_rows) + '''

![Controlled contact interventions](assets/coupling.png)

**해석:** 실제 simulator의 결합량은 0이 아닙니다. 이는 접촉 두 개의 반응을 독립적으로 더하는 것만으로는 설명되지 않는 효과가 있음을 보여줍니다. 반면 3-epoch 모델의 예측 결합량은 작아 실제 결합을 충분히 재현하지 못했습니다. 이 smoke 모델로 aggregation 학습 성공을 주장할 수 없습니다. 개입 지지대가 훈련 분포와 다르다는 점도 추가 통제가 필요합니다.

## 5. Cube → 두 링크 전이

source는 budget 8·seed 0의 Cube 모델입니다. target은 두 링크이며 2 epochs, seed 0에서 비교했습니다. 새로운 관절의 decoder는 source에서 학습되지 않았으므로 구조 전이 시 target에서 학습하도록 처리합니다. 자세한 freeze 예외와 파라미터 목록은 각 `training.json`에 기록됩니다.

| 적응 방식 | target trajectories | 선속도 RMSE (m/s) | 관절 위치 RMSE (rad) |
|---|---:|---:|---:|
''' + "\n".join(transfer_rows) + f'''

![Transfer modes](assets/transfer.png)

**해석:** 네 가지 적응 경로가 실행됐습니다. 적은 epoch와 단일 seed이므로 encoder 재사용의 이득은 아직 검증되지 않았습니다. 본 실험에서는 4·16·64 trajectory와 여러 seed, 동일 물성 전이와 물성 변화 전이를 구분해야 합니다.

## 6. Ant dynamics와 사족 보행

Ant는 floating base와 8개 hinge를 가진 자체 MJCF 모델입니다. 4개 train trajectory, 128 transitions, 3 epochs로 학습했습니다. 독립 test에서 1-step 위치 RMSE **{evaluation['mean_one_step']['position_rmse']:.4f} m**, 선속도 RMSE **{evaluation['mean_one_step']['linear_velocity_rmse']:.4f} m/s**, 관절 위치 RMSE **{evaluation['mean_one_step']['joint_position_rmse']:.4f} rad**입니다. 16-step rollout의 수치 발산 비율은 {evaluation['divergence_rate']:.1%}이며, 발산이 없다는 사실만으로 물리 정확성을 뜻하지 않습니다.

| 제어기 | 평가 시간 | 전진 거리 | 평균 속도 | 낙상 프레임 비율 |
|---|---:|---:|---:|---:|
| PD diagonal trot | {ant['duration_seconds']:.1f}s | {ant['forward_displacement']:.3f}m | {ant['average_forward_speed']:.3f}m/s | {ant['fall_fraction']:.1%} |
| 무제어 | {zero['duration_seconds']:.1f}s | {zero['forward_displacement']:.3f}m | {zero['average_forward_speed']:.3f}m/s | {zero['fall_fraction']:.1%} |
| PD trot (동일 짧은 구간) | {short['duration_seconds']:.1f}s | {short['forward_displacement']:.3f}m | {short['average_forward_speed']:.3f}m/s | {short['fall_fraction']:.1%} |
| 학습 모델 MPC | {mpc['duration_seconds']:.1f}s | {mpc['forward_displacement']:.3f}m | {mpc['average_forward_speed']:.3f}m/s | {mpc['fall_fraction']:.1%} |

![Ant progress and height](assets/ant-control.png)

<video controls preload="metadata" poster="assets/ant-walking.png"><source src="assets/ant-walking.mp4" type="video/mp4"></video>

PD trot은 사람이 설계한 feedback gait이며 RL로 학습한 보행 정책이 아닙니다. MPC는 이 gait를 초기 action proposal로 사용하고 학습 모델의 비용으로 후보를 고릅니다. **성공한 보행 전체를 world model의 기여로 해석하면 안 됩니다.** 실제 MuJoCo에서 별도로 action을 실행해 평가했습니다. MPC는 horizon 3, 후보 8, iteration 1이며 평균 planning 시간 **{mpc['planning_mean_ms']:.1f} ms**입니다. 제어 간격 20 ms에 대한 실시간 제어 보장은 없습니다.

## 7. Cube 목표 위치 제어

같은 1초 구간에서 baseline의 최종 목표 거리 오차는 **{cube_baseline['final_goal_distance']:.3f}m**, 학습 모델 MPC는 **{cube_mpc['final_goal_distance']:.3f}m**였습니다. 각 성공 판정은 **{cube_baseline['success']} / {cube_mpc['success']}**입니다. smoke 학습의 외력 범위보다 제어 외력이 크므로 이 결과에는 action 분포 변화가 포함됩니다. 모델 성능 비교용 제어 실험에서는 같은 action 범위의 학습 데이터를 충분히 확보해야 합니다.

## 8. 남아 있는 연구 검증

- 현재 결과는 구현 확인입니다. 계층형 구조의 우월성과 few-shot 전이 이득은 미입증입니다.
- 원본 NeRD·FIGNet·RigidFormer 논문 성능을 재현한 결과는 없습니다. 내부 NeRD-like 비교군을 원본으로 표시하지 않습니다.
- 동일 파라미터·계산 예산 보정, 3개 이상의 seed, 충분한 학습 수렴, 64포인트/면 조건의 실험이 필요합니다.
- point density와 Hilbert 근접쌍 coverage를 확인한 뒤 국소 attention의 효과를 해석해야 합니다.
- contact mode probe·보조 mode loss는 본 구현의 기본 학습 경로에 포함되지 않습니다.
- 단일 world Warp adapter는 정확성 검증 중심입니다. GPU 병렬 수집 throughput 벤치마크는 별도로 수행해야 합니다.

## 재현 명령과 원시 결과

```bash
docker compose --profile gpu run --rm warp python -m cube quick --with-warp
docker compose run --rm -e MUJOCO_GL=osmesa research python -m cube render --trajectory outputs/quick/ant_baseline/rollout.npz --scene outputs/quick/ant_baseline/scene.json --output docs/assets/ant-walking.mp4 --stride 2
docker compose run --rm research python -m cube docs
```

GPU 없이 첫 명령을 `docker compose run --rm research python -m cube quick`로 실행하면 CPU 실험을 수행합니다. 결과는 `{output}`에 저장됩니다. JSON은 모델별·시나리오별 오차, 실제 parameter 수, seed, trajectory 목록, 전이 시 불러온 파라미터, 각 control step을 포함합니다. 그래프는 `cube report`로 원시 결과에서 다시 생성할 수 있습니다.
'''
    Path("documentation").mkdir(exist_ok=True)
    if (root / "pytest.xml").exists():
        suites = list(ET.parse(root / "pytest.xml").getroot().iter("testsuite"))
        totals = {key: sum(int(s.get(key, 0)) for s in suites) for key in ["tests", "failures", "errors", "skipped"]}
        passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
        text += f"\n## 자동 검증 기록\n\n별도 GPU-enabled pytest 실행: **{passed} passed**, {totals['failures']} failures, {totals['errors']} errors, {totals['skipped']} skipped. 상태 계약·실제 Warp physics·FK/속도 일치·rollout gradient·독립 합산·데이터 분리·전이 head를 포함합니다. 원시 결과는 `outputs/quick/pytest.xml`입니다.\n"
    text += "\n[실행 환경·소스 hash](results/run_metadata.json) · [Cube 비교 원시 결과](results/suite.json) · [전이 원시 결과](results/transfer.json) · [접촉 개입 원시 결과](results/coupling.json)\n"
    Path("documentation/10_results.md").write_text(text, encoding="utf-8")
    if (root / "coverage.json").exists():
        coverage, null = read("coverage.json"), read("independent_null.json")
        lines = []
        for row in coverage["summary"]:
            ratio = f"{row['weighted_coverage']:.1%}" if row["weighted_coverage"] is not None else "정의되지 않음 (근접쌍 0)"
            lines.append(f"| {row['profile']} | {row['serialization']} | {row['local_layers']} | {row['near_pairs']} | {ratio} |")
        diagnostic_text = '''# 표면 샘플링과 독립 합산 검증

## 무엇을 구현하고 왜 측정했는가

`scripts/inspect_coverage.py`는 실제 가까운 동적 표면–환경 포인트 쌍이 local attention window에 함께 들어가는지를 측정합니다. 상위 aggregation이 입력 단계에서 누락된 모든 접촉 정보를 복구할 수 있다고 가정하지 않기 위해서입니다.

Cube 정지 장면의 4개 seed에서 거리 0.08m 이하의 sampled pair를 검사했습니다. 비율은 여러 seed의 분자를 더한 뒤 분모의 합으로 나눈 값입니다. 1층은 unshifted, 3층은 unshifted와 반-window shifted 분할의 합집합입니다. 3층의 마지막 분할은 첫 층과 같으므로 direct-pair coverage는 2층과 같습니다.

| sampling profile | 정렬 | local 층 | 근접쌍 수 | 직접 attention coverage |
|---|---|---:|---:|---:|
''' + "\n".join(lines) + f'''

## 측정 결과를 반영한 설정

- `physical_density`: 면당 64·바닥 256·window 64.
- `smoke_density`: 면당 16·바닥 64·window 32. 최종 소규모 실험은 shifted local attention 2층을 사용합니다.
- `undersampled_density`: 면당 4·바닥 16. 실제 물리 접촉이 있어도 0.08m 이내의 sampled pair가 없습니다. window만 늘려도 샘플 자체의 누락을 해결할 수 없습니다.

이 검사 결과를 반영해 최종 smoke 설정의 표면 밀도를 높이고 모든 소규모 학습 결과를 다시 측정했습니다. 이는 무작정 모델 용량을 늘리기 전에 입력 접촉 표현을 검사해야 하는 이유입니다. 여기서 coverage 100%는 검사한 4개 정지 상태에만 해당하며, 모든 운동·형상에 대한 보장이 아닙니다.

## 독립 합산 비교군의 음성 대조

`independent`는 각 static geometry의 반응을 독립적으로 계산한 뒤 더합니다. 동일 상태에서 A/B/AB/∅를 바꾸는 8가지 검사에서 예측 world-twist 결합 norm의 최댓값은 **{null['max_predicted_world_twist_coupling_norm']:.3g}**, 허용오차는 {null['tolerance']:.3g}였고 **통과={null['passed']}**입니다.

이 검사는 학습 전 초기화된 모델의 구조적 합산 성질을 검증합니다. 예측 정확성이 좋다는 뜻이 아니며, 결합을 표현할 수 없는 비교군이 의도대로 제한됐는지 확인합니다. 선속도·각속도를 합친 norm은 수치적 0 검사에만 사용합니다.

```bash
docker compose run --rm research python scripts/inspect_coverage.py --output outputs/quick
docker compose run --rm research python -m cube report
docker compose run --rm research python -m cube docs
```

[Coverage 원시 JSON](results/coverage.json) · [독립 합산 원시 JSON](results/independent_null.json)
'''
        Path("documentation/11_sampling_validation.md").write_text(diagnostic_text, encoding="utf-8")
    result_dir = Path("docs/results")
    result_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["run_metadata.json", "coverage.json", "independent_null.json"]:
        if (root / filename).exists():
            shutil.copyfile(root / filename, result_dir / filename)
    for filename in ["suite/suite.json", "transfer/transfer.json", "coupling_model/coupling.json"]:
        shutil.copyfile(root / filename, result_dir / Path(filename).name)
    return {"source": "documentation/10_results.md", "figures": str(assets), "runs": len(suite["runs"])+len(transfer["runs"])+1}
