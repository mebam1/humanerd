"""Build Korean charts and an auditable summary of complete OOD v2 evaluations.

Run after scripts/evaluate_ood_v2.py has produced all four evaluation.json files.
This script reads existing model checkpoints and evaluation data. It neither trains
models nor logs anything to W&B. Example (Malgun font mounted read-only):

    python scripts/build_ood_v2_charts.py --font /fonts/malgun.ttf
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import torch


MODELS = ("state_mlp", "deepsets", "hierarchical", "contact_mlp")
MODEL_NAMES = {"state_mlp": "State MLP", "deepsets": "DeepSets", "hierarchical": "Hierarchical", "contact_mlp": "Contact MLP"}
COLORS = {"state_mlp": "#2b82b9", "deepsets": "#10887c", "hierarchical": "#8b5dba", "contact_mlp": "#c57624"}
SECTIONS = ("rollout", "one_step", "impact", "persistent_contact")
SCENARIOS = ("toss_wall", "toss_valley")
METRICS = {
    "position_rmse": ("위치 RMSE", "m"),
    "rotation_geodesic_rad": ("회전 지오데식 오차", "rad"),
    "linear_velocity_rmse": ("선속도 RMSE", "m/s"),
    "angular_velocity_rmse": ("각속도 RMSE", "rad/s"),
}
SECTION_NAMES = {
    "rollout": "100스텝 자율 예측",
    "one_step": "한 스텝 예측",
    "impact": "충돌 구간 한 스텝 예측",
    "persistent_contact": "지속 접촉 구간 한 스텝 예측",
}
SECTION_EXPLANATIONS = {
    "rollout": "최초 정답 상태에서 시작해 자신의 예측을 최대 100스텝(2초) 이어 붙였을 때의 누적 상태 오차입니다.",
    "one_step": "매 순간 정답 상태로 다음 한 스텝을 예측하여 오차 누적 없이 낯선 장면에 대한 즉각적인 반응을 봅니다.",
    "impact": "한 스텝 예측 중 정답의 충돌 이벤트가 표시된 프레임만 골라 충돌 순간의 상태 오차를 봅니다.",
    "persistent_contact": "한 스텝 예측 중 정답의 접촉이 연속되는 프레임만 골라 지속 접촉 중의 상태 오차를 봅니다.",
}
SCENARIO_NAMES = {"overall": "전체 OOD 1,000개", "toss_wall": "벽 투척 500개", "toss_valley": "V자 바닥 500개"}
SCENARIO_EXPLANATIONS = {
    "toss_wall": "랜덤 위치·회전에서 평평한 벽을 향해 던진 큐브가 벽과 평평한 바닥에 충돌하는 반응을 평가합니다.",
    "toss_valley": "학습과 같은 초기 투척 분포에서 각 평면이 수평보다 10° 기울어진 두 바닥의 낮은 중앙과 접촉하는 반응을 평가합니다.",
}
DIAGNOSTICS = (
    "penetration_mean", "penetration_max", "contact_disagreement", "contact_onset_error_seconds",
    "missed_contact_onsets", "false_contact_onsets",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def local_path(root: Path, path: str) -> Path:
    """Accept workspace-relative paths or paths recorded from a /workspace mount."""
    if path.startswith("/workspace/"):
        return root / path[len("/workspace/"):]
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def source_record(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def nullable_mean(values):
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def summarize(records: dict[str, list[dict]], horizon: int) -> dict:
    values, counts = {}, {}
    for section in SECTIONS:
        values[section] = {}
        for model in MODELS:
            eligible = [record[section] for record in records[model] if record[section] is not None]
            counts.setdefault(model, {})[section] = len(eligible)
            values[section][model] = {
                metric: nullable_mean([record[metric] for record in eligible]) for metric in METRICS
            }
    reductions = {
        section: {
            model: {
                metric: (100 * (1 - values[section][model][metric] / values[section]["state_mlp"][metric])
                         if values[section][model][metric] is not None
                         and values[section]["state_mlp"][metric] not in (None, 0) else None)
                for metric in METRICS
            }
            for model in MODELS if model != "state_mlp"
        }
        for section in SECTIONS
    }
    diagnostics = {
        model: {key: nullable_mean([record[key] for record in records[model]]) for key in DIAGNOSTICS}
        for model in MODELS
    }
    diagnostic_counts = {
        model: {key: sum(record[key] is not None for record in records[model]) for key in DIAGNOSTICS}
        for model in MODELS
    }
    rollout_steps = {}
    distributions = {}
    for model in MODELS:
        steps = [record["rollout_steps"] for record in records[model]]
        rollout_steps[model] = {
            "requested": horizon, "min": min(steps), "max": max(steps), "mean": mean(steps),
            "completed_horizon_count": sum(step == horizon for step in steps),
            "truncated_count": sum(step < horizon for step in steps),
            "diverged_count": sum(record["diverged"] for record in records[model]),
        }
        distributions[model] = {}
        for metric in METRICS:
            sample = [record["rollout"][metric] for record in records[model] if record["rollout"] is not None]
            distributions[model][metric] = {
                "median": float(np.median(sample)) if sample else None,
                "p95": float(np.quantile(sample, .95, method="linear")) if sample else None,
                "max": max(sample) if sample else None,
                "min": min(sample) if sample else None,
                "count": len(sample),
            }
    return {
        "mean_errors": values, "reduction_percent_vs_state_mlp": reductions,
        "metric_trajectory_counts": counts, "diagnostics": diagnostics, "diagnostic_trajectory_counts": diagnostic_counts,
        "divergence_rate": {model: mean(record["diverged"] for record in records[model]) for model in MODELS},
        "rollout_steps": rollout_steps, "rollout_distribution": distributions,
        "event_frames": {
            model: {"impact": sum(record["impact_frames"] for record in records[model]),
                    "persistent_contact": sum(record["persistent_frames"] for record in records[model])}
            for model in MODELS
        },
    }


def configure_font(path: str | None):
    candidates = [Path(path)] if path else [Path("/fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgun.ttf")]
    selected = next((candidate for candidate in candidates if candidate.is_file()), None)
    if selected is None:
        raise FileNotFoundError("A Korean font is required; pass --font /path/to/malgun.ttf")
    font_manager.fontManager.addfont(str(selected))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(selected)).get_name(),
        "axes.unicode_minus": False, "svg.fonttype": "path", "font.size": 11,
        "axes.labelcolor": "#516984", "xtick.color": "#516984", "ytick.color": "#516984",
        "text.color": "#142740", "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": "#d9e3ec",
    })


def display(value):
    if value is None:
        return "자료 없음"
    return f"{value:.5f}" if value < .01 else f"{value:.4f}"


def chart(destination: Path, scope: str, section: str, summary: dict, count: int) -> dict:
    basename = f"{scope}-{section.replace('_', '-')}-errors"
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.3))
    fig.subplots_adjust(left=.08, right=.975, bottom=.16, top=.80, wspace=.28, hspace=.48)
    fig.text(.08, .95, f"{SCENARIO_NAMES[scope]} · {SECTION_NAMES[section]}", fontsize=21, fontweight="bold")
    fig.text(.08, .901, SECTION_EXPLANATIONS[section], fontsize=10.9, color="#516984")
    fig.text(.08, .859, "OOD v2 로컬 재평가 · 낮을수록 좋음 · 각 패널은 0에서 시작하는 독립 선형 세로축",
             fontsize=10.5, color="#516984")
    for ax, (metric, (label, unit)) in zip(axes.flat, METRICS.items()):
        heights = [summary["mean_errors"][section][model][metric] for model in MODELS]
        finite = [value for value in heights if value is not None]
        maximum = max(finite, default=0.) or 1.
        bars = ax.bar(range(len(MODELS)), [value or 0 for value in heights],
                      color=[COLORS[model] for model in MODELS], width=.57, zorder=3)
        ax.set_xticks(range(len(MODELS)), [MODEL_NAMES[model] for model in MODELS], fontsize=10)
        ax.set_title(label, loc="left", fontsize=13.5, fontweight="bold", pad=12)
        ax.set_ylabel(unit, rotation=0, ha="right", labelpad=10)
        ax.set_ylim(0, maximum * 1.28)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=4))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(axis="y", color="#e5edf3", zorder=0)
        ax.tick_params(axis="both", length=0, pad=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for bar, value, model in zip(bars, heights, MODELS):
            ax.text(bar.get_x() + bar.get_width()/2, (value or 0) + maximum*.045, display(value),
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=COLORS[model])
    eligible = [summary["metric_trajectory_counts"][model][section] for model in MODELS]
    count_note = str(eligible[0]) if len(set(eligible)) == 1 else "/".join(map(str, eligible))
    fig.text(.08, .090, f"동일한 {count:,}개 궤적 비교 · 지표 산출 궤적 {count_note}개 · 궤적별 오차의 산술평균 · 5 epoch / seed 0",
             fontsize=10.1, color="#516984")
    if section == "rollout":
        completed = [summary["rollout_steps"][model]["completed_horizon_count"] for model in MODELS]
        detail = ("네 모델의 전 궤적 100스텝 완료." if completed == [count]*len(MODELS)
                  else "100스텝 완료: " + ", ".join(f"{MODEL_NAMES[model]} {n}" for model, n in zip(MODELS, completed)) + "개. 조기 종료는 완료 구간 오차.")
    else:
        detail = "정답 상태 입력의 한 스텝 평가이며, 충돌·지속 접촉은 정답 이벤트 마스크로 선택합니다."
    fig.text(.08, .051, detail, fontsize=9.7, color="#516984")
    paths = {extension: f"{basename}.{extension}" for extension in ("png", "svg")}
    fig.savefig(destination / paths["png"], dpi=160)
    fig.savefig(destination / paths["svg"])
    plt.close(fig)
    return paths


def build(root: Path, evaluation_dir: Path, dataset_dir: Path, destination: Path, horizon: int):
    manifest_path = dataset_dir / "manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["status"] == "complete" and manifest["ood_only"]
    assert manifest["config"]["steps"] == horizon == 100
    entries, shard_sources = list(manifest["trajectories"]), []
    if not entries:
        for shard in manifest["shards"]:
            metadata_path = dataset_dir / shard["path"] / "metadata.json"
            metadata = read_json(metadata_path)
            assert metadata["dataset_fingerprint"] == manifest["dataset_fingerprint"]
            entries.extend(metadata["trajectories"])
            shard_sources.append(source_record(root, metadata_path))
    entries = [entry for entry in entries if entry["split"] == "ood"]
    counts = dict(Counter(entry["scenario"] for entry in entries))
    assert counts == {"toss_wall": 500, "toss_valley": 500}, counts
    assert len({entry["id"] for entry in entries}) == len(entries) == 1000
    entry_by_id = {entry["id"]: entry for entry in entries}
    reports, records, sources, checkpoints, training_reports = {}, {}, {}, {}, {}
    for model in MODELS:
        report_path = evaluation_dir / model / "evaluation.json"
        report = read_json(report_path)
        assert report["complete"] and report["model_name"] == model
        assert report["split"] == "ood"
        assert local_path(root, report["dataset"]).resolve() == dataset_dir.resolve()
        assert report["dataset_fingerprint"] == manifest["dataset_fingerprint"]
        assert report["dataset_manifest_sha256"] == sha256(manifest_path)
        numerical_checks = report["batch_equivalence_checks"]
        assert {check["scenario"] for check in numerical_checks} == set(SCENARIOS)
        assert all(check["one_step_comparison_passed"] for check in numerical_checks)
        assert report["available_trajectories"] == report["evaluated_trajectories"] == 1000
        assert set(report["trajectory_ids"]) == set(entry_by_id)
        assert len(report["trajectory_ids"]) == len(set(report["trajectory_ids"]))
        rows = report["trajectories"]
        if not rows:
            rows = [json.loads(line) for line in (report_path.parent / "trajectories.jsonl").read_text().splitlines()]
        assert len(rows) == 1000 and [row["trajectory_id"] for row in rows] == report["trajectory_ids"]
        for row in rows:
            assert row["scenario"] == entry_by_id[row["trajectory_id"]]["scenario"]
            assert 0 <= row["rollout_steps"] <= horizon
            assert row["rollout_steps"] == horizon or row["diverged"]
            for section in SECTIONS:
                if row[section] is not None:
                    assert all(math.isfinite(row[section][key]) and row[section][key] >= 0 for key in METRICS)
        checkpoint_path = local_path(root, report["checkpoint"])
        assert report["checkpoint_sha256"] == sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 5 and checkpoint["seed"] == 0
        assert checkpoint["batch_cursor"] == 0 and checkpoint["next_epoch"] == 5
        assert checkpoint["model_config"]["name"] == model
        training_path = (root / "outputs/contact-mlp-5ep/training.json" if model == "contact_mlp" else
                         root / f"outputs/cube-toss-suite-final/{model}_budget10000_seed0/training.json")
        training = read_json(training_path)
        assert len(training["epochs"]) == 5 and all(row["epoch_complete"] for row in training["epochs"])
        assert training["dataset_fingerprint"] == checkpoint["dataset_fingerprint"]
        assert training["normalization"] == checkpoint["normalization"]
        checkpoints[model] = {
            **source_record(root, checkpoint_path), "epoch": checkpoint["epoch"], "seed": checkpoint["seed"],
            "global_step": checkpoint["global_step"], "training_dataset_fingerprint": checkpoint["dataset_fingerprint"],
            "model_config": checkpoint["model_config"], "training_trajectories": training["train_trajectories"],
            "validation_trajectories": training["val_trajectories"],
            "target_normalization_matches_state_mlp": True,
            "input_representation": ("analytic eight-corner explicit contact slots; body-frame motion/wrench/gravity"
                                     if model == "contact_mlp" else "existing point-cloud geometry context"),
            "input_normalization": training.get("input_normalization"),
        }
        sources[model] = source_record(root, report_path)
        sources[model]["trajectory_metrics"] = source_record(root, report_path.parent / "trajectories.jsonl")
        sources[model]["training_report"] = source_record(root, training_path)
        reports[model], records[model], training_reports[model] = report, rows, training
    for model in MODELS:
        assert reports[model]["trajectory_ids"] == reports["state_mlp"]["trajectory_ids"]
        assert training_reports[model]["normalization"] == training_reports["state_mlp"]["normalization"]
        assert training_reports[model]["dataset_fingerprint"] == training_reports["state_mlp"]["dataset_fingerprint"]
    result = summarize(records, horizon)
    for model in MODELS:
        for section in SECTIONS:
            for metric in METRICS:
                assert math.isclose(result["mean_errors"][section][model][metric],
                                    reports[model][f"mean_{section}"][metric], rel_tol=1e-10, abs_tol=1e-12)
            if section != "one_step":
                assert result["metric_trajectory_counts"][model][section] == reports[model]["metric_trajectory_counts"][section]
        assert result["divergence_rate"][model] == reports[model]["divergence_rate"]
        for key in DIAGNOSTICS:
            actual, recorded = result["diagnostics"][model][key], reports[model]["mean_diagnostics"][key]
            assert actual == recorded if actual is None or recorded is None else math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-12)
    scenarios = {
        scenario: {"trajectory_count": counts[scenario],
                   **summarize({model: [row for row in records[model] if row["scenario"] == scenario]
                                for model in MODELS}, horizon)}
        for scenario in SCENARIOS
    }
    for scenario in SCENARIOS:
        for model in MODELS:
            recorded = reports[model]["scenarios"][scenario]
            assert recorded["evaluated_trajectories"] == counts[scenario]
            for section in SECTIONS:
                for metric in METRICS:
                    observed = scenarios[scenario]["mean_errors"][section][model][metric]
                    expected = recorded[f"mean_{section}"].get(metric)
                    assert observed == expected if observed is None or expected is None else math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-12)
    dt = manifest["config"]["scene"]["dt"]
    result.update({
        "schema_version": 2, "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "builder_source": source_record(root, Path(__file__).resolve()),
        "source": "Local evaluation of the existing 5-epoch seed-0 checkpoints on all 1,000 OOD v2 trajectories",
        "remote_verified": False, "remote_writes": False, "wandb_logging": False,
        "models": list(MODELS), "model_names": MODEL_NAMES,
        "metric_names": {key: value[0] for key, value in METRICS.items()},
        "metric_units": {key: value[1] for key, value in METRICS.items()},
        "section_names": SECTION_NAMES, "section_explanations_ko": SECTION_EXPLANATIONS,
        "scenario_names": {key: SCENARIO_NAMES[key] for key in SCENARIOS},
        "scenario_explanations_ko": SCENARIO_EXPLANATIONS,
        "metric_explanations_ko": {
            "position_rmse": "큐브 중심 위치의 x·y·z 성분 오차를 제곱해 평균한 뒤 제곱근을 취한 값입니다(m).",
            "rotation_geodesic_rad": "예측 자세에서 정답 자세까지의 최단 회전각을 평균한 값입니다(rad).",
            "linear_velocity_rmse": "큐브 중심 선속도의 x·y·z 성분 오차에 대한 RMSE입니다(m/s).",
            "angular_velocity_rmse": "월드 좌표계 각속도의 x·y·z 성분 오차에 대한 RMSE입니다(rad/s).",
        },
        "scenarios": scenarios,
        "numerical_validation": {
            model: reports[model].get("batch_equivalence_checks", []) for model in MODELS
        },
        "trajectory_counts": {"total": 1000, "scenarios": counts,
                              "per_model": {model: len(records[model]) for model in MODELS}},
        "protocol": {
            "split": "ood", "evaluated_trajectories": 1000, "available_trajectories": 1000,
            "scenario_counts": counts, "horizon_steps": horizon, "dt_seconds": dt,
            "horizon_seconds": horizon*dt, "checkpoint_epoch": 5, "training_seed": 0,
            "training_trajectories": training_reports["state_mlp"]["train_trajectories"],
            "validation_trajectories": training_reports["state_mlp"]["val_trajectories"],
            "all_models_same_trajectory_ids": True, "trajectory_ids": reports["state_mlp"]["trajectory_ids"],
            "all_models_same_target_normalization": True, "wandb_logging": False,
            "input_normalization_note_ko": "contact_mlp는 학습 데이터만으로 구한 자체 입력 정규화를 사용합니다. 손실 함수와 타깃 변화량 정규화는 네 모델이 같습니다.",
            "evaluation_engine_by_model": {model: reports[model].get("evaluation_engine") for model in MODELS},
            "trajectory_batch_size_by_model": {model: reports[model].get("trajectory_batch_size") for model in MODELS},
            "one_step_batch_size_by_model": {model: reports[model].get("one_step_batch_size") for model in MODELS},
            "numerical_note_ko": "같은 장면의 궤적을 FP32 배치로 평가했으며, 100스텝에서는 배치·순차 연산의 반올림 차이가 누적될 수 있습니다. 파일에 파일럿의 한 스텝·100스텝 비교 기록을 보존했습니다.",
            "aggregate_definition": "arithmetic mean of per-trajectory errors; not pooled RMSE and not final-frame error",
            "rollout_definition": "ground-truth initial state followed by autonomous model predictions for up to 100 steps",
            "one_step_definition": "ground-truth state input for each predicted transition",
            "event_filter_definition": "one-step predictions masked by GT impact_event or persistent_contact labels",
            "early_termination_definition": "nonfinite state or max absolute state > 1e4; metrics use finite completed prefix only",
            "quantile_definition": "numpy.quantile(method='linear') on per-trajectory rollout metrics",
        },
        "checkpoints": checkpoints, "local_sources": sources,
        "dataset_identity": {
            "path": dataset_dir.relative_to(root).as_posix(), "version": "OOD v2",
            "fingerprint": manifest["dataset_fingerprint"], "manifest": source_record(root, manifest_path),
            "shard_metadata": shard_sources, "seed": manifest["seed"],
            "schema_hash": manifest["schema_hash"], "generator_version": manifest["generator_version"],
            "backend": manifest["backend"], "rigid_action_mode": manifest["rigid_action_mode"],
            "valley_angle_degrees_per_plane": manifest["config"]["scene"]["valley_angle_degrees"],
        },
        "evaluation_summaries": {
            model: {key: value for key, value in reports[model].items() if key not in ("trajectories", "trajectory_ids")}
            for model in MODELS
        },
        "limitations_ko": [
            "단일 학습 seed의 5 epoch 체크포인트 비교이며 여러 seed의 신뢰구간을 평가하지 않았습니다.",
            "이번 OOD는 평평한 벽과 각 10°의 두 평면으로 제한되며, 임의 구덩이·융기·곡면에서의 일반화는 추가 검증이 필요합니다.",
            "contact_mlp는 고정 꼭짓점 8개의 해석적 접촉 입력을 사용합니다. point 기반 모델과 입력 구성은 다르므로, 네트워크 구조만의 효과를 분리한 ablation이 아닙니다.",
            "rollout 조기 종료가 있으면 해당 궤적 오차는 완료된 유한 프레임까지만의 값이므로 완주 개수와 함께 해석해야 합니다.",
            "침투와 접촉 불일치는 예측 상태를 시뮬레이터로 진단한 별도 물리 지표이며 모델 입력으로 제공하지 않았습니다.",
            "이 수치는 새 OOD v2 데이터의 로컬 재평가 결과이며 기존 W&B run의 OOD 기록을 갱신하지 않았습니다.",
            "FP32 배치 자율 예측의 반올림 차이가 장기 접촉 궤적에서 증폭될 수 있으며, 기존 순차 평가와의 파일럿 수치 비교는 numerical_validation에 기록했습니다.",
        ],
    })
    audit_path = root / "docs/results/ood_toss_v2_audit.json"
    audit = read_json(audit_path)
    assert audit["dataset_fingerprint"] == manifest["dataset_fingerprint"] and audit["mixture_50_50_valid"]
    result["dataset_identity"]["audit"] = source_record(root, audit_path)
    result["dataset_identity"]["array_sources"] = [source_record(root, array)
                                                   for shard in manifest["shards"]
                                                   for array in sorted((dataset_dir / shard["path"]).glob("*.npy"))]
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for scope, summary, count in [("overall", result, 1000), *[(scenario, scenarios[scenario], counts[scenario]) for scenario in SCENARIOS]]:
        files[scope] = {section: chart(destination, scope, section, summary, count) for section in SECTIONS}
    result["files"] = files
    result["file_sha256"] = {name: sha256(destination / name)
                            for scope in files.values() for group in scope.values() for name in group.values()}
    (destination / "ood-data.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"destination": str(destination), "figures": len(files)*len(SECTIONS),
                      "trajectory_counts": result["trajectory_counts"], "divergence_rate": result["divergence_rate"],
                      "rollout_steps": result["rollout_steps"],
                      "rollout_reduction_percent": result["reduction_percent_vs_state_mlp"]["rollout"]}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evaluation-dir", default="outputs/ood-v2-evaluation")
    parser.add_argument("--dataset", default="data/cube-toss-ood-v2")
    parser.add_argument("--output", default="docs/assets/ood-v2-comparison")
    parser.add_argument("--font")
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    configure_font(args.font)
    root = args.root.resolve()
    build(root, local_path(root, args.evaluation_dir), local_path(root, args.dataset),
          local_path(root, args.output), args.horizon)


if __name__ == "__main__":
    main()
