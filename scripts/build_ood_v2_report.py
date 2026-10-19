"""Build the Korean OOD v2 report from verified local evaluation summaries.

Run after scripts/build_ood_v2_charts.py has produced the JSON and figures.
The report embeds its data so its selectors also work when opened via file://.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path
import re
from string import Template

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9), "Asia/Seoul")
MODELS = ("state_mlp", "deepsets", "hierarchical", "contact_mlp")
FIELDS = (
    ("position_rmse", "위치 RMSE", "m"),
    ("rotation_geodesic_rad", "회전 거리 평균", "rad"),
    ("linear_velocity_rmse", "선속도 RMSE", "m/s"),
    ("angular_velocity_rmse", "각속도 RMSE", "rad/s"),
)
GROUPS = {
    "rollout": ("100스텝 자율 rollout", "정답 상태로 교체하지 않고 자신의 예측을 2초 동안 다음 입력으로 사용해, 미지 환경에서 오차가 얼마나 누적되는지 확인합니다."),
    "one_step": ("한 스텝 예측", "매 시점 정답 상태에서 다음 순간을 예측해, 처음 만나는 표면에서 즉각적인 상태 변화를 읽는지 확인합니다."),
    "impact": ("충돌 순간", "GT에서 충돌로 표시된 시점의 한 스텝 오차를 모아, 미경험 벽과 바닥에 부딪힐 때 반사·회전을 예측하는지 확인합니다."),
    "persistent_contact": ("지속 접촉", "GT에서 접촉이 유지되는 시점의 한 스텝 오차를 모아, 새로운 표면의 지지·이동·회전을 예측하는지 확인합니다."),
}
SCOPES = {"overall": "전체 · 1,000개", "toss_wall": "벽 투척 · 500개", "toss_valley": "V자 바닥 · 500개"}


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value, digits=5):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def reduction_text(value):
    if value is None:
        return "비교 불가"
    if abs(value) < .05:
        return "차이 0.0%"
    return f"{abs(value):.1f}% {'감소' if value > 0 else '증가'}"


def summary_note(data):
    errors = data["mean_errors"]["rollout"]
    baseline = errors["state_mlp"]["position_rmse"]
    return "전체 1,000개의 100스텝 rollout 위치 RMSE는 " + ", ".join(
        f"{model} {errors[model]['position_rmse']:.5f} m" for model in MODELS
    ) + ". state_mlp 대비 변화는 " + ", ".join(
        model + " " + reduction_text((1 - errors[model]["position_rmse"] / baseline) * 100)
        for model in MODELS[1:]
    ) + "입니다."


def scenario_note(data):
    sentences = []
    for model, label in (("deepsets", "DeepSets"), ("hierarchical", "hierarchical"), ("contact_mlp", "Contact MLP")):
        clauses = []
        reductions = []
        for scenario, name in (("toss_wall", "벽 투척"), ("toss_valley", "V자 바닥")):
            values = data["scenarios"][scenario]["mean_errors"]["rollout"]
            base = values["state_mlp"]["position_rmse"]
            value = values[model]["position_rmse"]
            change = (1 - value / base) * 100
            reductions.append(change)
            clauses.append(f"{name}에서 {reduction_text(change)}({base:.5f} → {value:.5f} m)")
        common = "두 조건 모두 state_mlp보다 낮습니다: " if all(v > 0 for v in reductions) else "조건에 따라 다릅니다: " if reductions[0] * reductions[1] < 0 else "다음과 같습니다: "
        sentences.append(label + "의 100스텝 위치 RMSE는 " + common + ", ".join(clauses) + ".")
    return " ".join(sentences)


def table_rows(data, group="rollout"):
    errors = data["mean_errors"][group]
    baseline = errors["state_mlp"]
    rows = []
    for model in MODELS:
        cells = []
        for key, _, unit in FIELDS:
            value = errors[model].get(key)
            base = baseline.get(key)
            change = (1 - value / base) * 100 if value is not None and base else None
            note = "기준 모델" if model == "state_mlp" else "state_mlp 대비 " + reduction_text(change)
            cells.append(f"<td><strong>{fmt(value)}</strong> {unit}<small>{note}</small></td>")
        rows.append(f'<tr><th scope="row">{model}</th>{"".join(cells)}</tr>')
    return "".join(rows)


def diagnostic_rows(data):
    rows = []
    for model in MODELS:
        d = data["diagnostics"][model]
        steps = data["rollout_steps"][model]
        rows.append(f'<tr><th scope="row">{model}</th><td>{data["divergence_rate"][model] * 100:.1f}%</td>'
                    f'<td>{steps["completed_horizon_count"]} / {steps["completed_horizon_count"] + steps["truncated_count"]}</td>'
                    f'<td>{steps["min"]}–{steps["max"]}<small>평균 {steps["mean"]:.1f}</small></td>'
                    f'<td>{fmt(d.get("penetration_mean"))} m</td><td>{fmt(d.get("penetration_max"))} m</td>'
                    f'<td>{fmt(d.get("contact_disagreement", 0) * 100, 2)}%</td></tr>')
    return "".join(rows)


def distribution_rows(data):
    rows = []
    for model in MODELS:
        d = data.get("rollout_distribution", {}).get(model, {}).get("position_rmse", {})
        rows.append(f'<tr><th scope="row">{model}</th><td>{fmt(d.get("median"))}</td><td>{fmt(d.get("p95"))}</td><td>{fmt(d.get("max"))}</td></tr>')
    return "".join(rows)


def read_page_style(docs):
    source = docs / "ood_comparison_v1.html"
    if not source.exists():
        source.write_bytes((docs / "ood_comparison.html").read_bytes())
    match = re.search(r"<style>(.*?)</style>", source.read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError("The archived v1 report must contain the shared style")
    return match.group(1)


def relative_artifact(value):
    value = str(value).replace("\\", "/")
    if value.startswith("docs/"):
        return value[5:]
    if value.startswith("assets/"):
        return value
    return "assets/ood-v2-comparison/" + value


def update_related(docs, data):
    note = ('<div class="note"><strong>새 OOD v2 평가 완료:</strong> '
            '<strong>벽 투척 500개 + 각 평면 10°인 V자 바닥 500개</strong>에서 5 epoch·seed 0의 네 모델을 비교했습니다. '
            '명시적 접촉 입력을 받는 <strong>contact_mlp</strong>도 추가했습니다. '
            + summary_note(data) + ' ' + scenario_note(data) + ' <a href="ood_comparison.html" target="_blank" rel="noopener">새 OOD v2 막대 그래프·조건별 비교</a>에서 확인하세요. '
            '이는 로컬 재평가이며 기존 W&amp;B의 <a href="ood_comparison_v1.html" target="_blank" rel="noopener">v1 기록</a>과 구분합니다. '
            '최종 목표는 단순 경사 변화가 아닌 <strong>구덩이·융기 같은 비평면 지형과 미경험 장애물</strong>에도 반응하는 NeRD입니다. '
            '<a href="nerd_goal.html" target="_blank" rel="noopener">연구 목표 보기</a></div>')
    for path in (docs / "deepsets_explained.html", ROOT / "outputs/build_deepsets_explained.py"):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text, count = re.subn(r'<div class="note"><strong>(?:학습에 없던 환경에서는\?|새 OOD v2 평가 완료:)</strong>.*?</div>', lambda _: note, text, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f"Could not find the OOD callout in {path}")
        path.write_text(text, encoding="utf-8")

    path = docs / "ood_comparison_v1.html"
    text = path.read_text(encoding="utf-8")
    archive_note = ('<p class="note"><strong>이 페이지는 이전 OOD v1 평가 기록입니다.</strong> '
                    '<strong>새 OOD v2 모델 평가 완료:</strong> 벽 투척 500개와 각 평면 10°인 V자 바닥 500개에서 '
                    'contact_mlp를 포함한 네 모델을 평가했습니다. <a href="ood_comparison.html">새 OOD v2 모델 오차·조건별 비교</a>와 '
                    '<a href="visualizations/ood-context/index.html">OOD v2 GT 갤러리</a>를 확인하세요. '
                    '아래 W&amp;B v1 지표와 본문은 이전 평가 기록이며 새 v2의 성능을 뜻하지 않습니다.</p>')
    text, count = re.subn(r'(<main>)<p class="note"><strong>이 페이지는 이전 OOD v1 평가 기록입니다\.</strong>.*?</p>', lambda m: m.group(1) + archive_note, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError("Could not find archived v1 status notice")
    path.write_text(text, encoding="utf-8")

    path = docs / "nerd_goal.html"
    text = path.read_text(encoding="utf-8")
    replacements = {
        'v2: GT 생성 · 모델 재평가 전': 'v2: 3개 모델 · 1,000개 재평가 완료',
        '<strong>v2는 데이터와 정답(GT) 영상 생성 단계</strong>이며 새 조건의 모델 평가를 아직 수행하지 않았습니다. 기존 W&amp;B 막대 그래프는 이전 <code>floor_wall</code> v1의 512개 평가 기록으로 남아 있습니다.': '<strong>v2의 GT와 세 모델의 1,000개 평가를 완료했습니다.</strong> <a href="ood_comparison.html">새 조건의 전체·벽·V자 바닥 오차와 발산 진단</a>을 확인할 수 있습니다. 기존 W&amp;B 막대 그래프는 <a href="ood_comparison_v1.html">이전 <code>floor_wall</code> v1의 512개 기록</a>으로 보존했습니다.',
        '이전 OOD v1의 모델 오차 비교와 v2 평가 범위 보기 →': '새 OOD v2의 세 모델 오차와 조건별 결과 보기 →',
        '기존 state_mlp, DeepSets, hierarchical 수치는 floor_wall v1의 결과입니다. 새 벽 투척·V자 바닥의 성능 수치와 구분해 확인할 수 있습니다.': 'state_mlp, DeepSets, hierarchical의 같은 5 epoch 체크포인트를 벽 투척 500개와 V자 바닥 500개에서 비교했습니다. 기존 v1 기록은 별도 보관합니다.',
        '100스텝, 2초 · 모델 예측·성능 평가 전': '100스텝, 2초 · GT 영상 / 모델 오차는 별도 비교',
        '모델의 반응과 오차는 새 데이터로 별도 평가해야 합니다.': '모델의 반응과 오차는 <a href="ood_comparison.html">새 데이터의 1,000개 평가</a>에서 확인할 수 있습니다.',
        '새 OOD v2 두 조건은 GT를 준비했고 모델 평가가 남아 있습니다.': '새 OOD v2 두 조건은 GT 생성과 세 모델의 전체 1,000개 평가를 완료했습니다.',
        '500개 GT 생성<br>모델 평가 전': '500개 GT 생성<br>세 모델 평가 완료',
        '기존 바닥 + 벽 OOD v1 모델 평가와 새 벽 투척·V자 바닥 OOD v2 GT를 구분합니다. 새 GT를 만들었다는 사실만으로 v2의 모델 성능이나 임의 구덩이·융기에서의 성공을 주장하지 않습니다.': '기존 바닥 + 벽 OOD v1 기록과 새 벽 투척·V자 바닥 OOD v2의 모델 평가를 구분합니다. v2의 실제 수치와 발산·접촉 진단을 함께 보고, 임의 구덩이·융기에서의 성공은 앞으로 검증합니다.',
        '이전 OOD v1 모델 평가와 새 평가 범위': '새 OOD v2 모델 평가와 조건별 비교',
        '기존 W&amp;B의 v1 OOD 평가 수치는 <a href="ood_comparison.html">OOD 비교 문서</a>에서 원본 기록과 함께 확인할 수 있습니다. v2 모델 평가는 아직 수행하지 않았습니다.': '새 OOD v2의 모델 수치는 <a href="ood_comparison.html">OOD 비교 문서</a>의 로컬 재평가 결과입니다. 기존 W&amp;B의 v1 수치는 <a href="ood_comparison_v1.html">v1 원본 기록</a>에서 확인할 수 있습니다.',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    for old, new in {
        '3개 모델 · 1,000개': '4개 모델 · 1,000개',
        '세 모델의': '네 모델의', '세 모델 평가 완료': '네 모델 평가 완료',
        'v2의 GT와 세 모델의': 'v2의 GT와 네 모델의',
        'state_mlp, DeepSets, hierarchical의': 'state_mlp, DeepSets, hierarchical, contact_mlp의',
        'v2의 세 모델 오차': 'v2의 네 모델 오차',
    }.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")

    path = docs / "visualizations/ood-context/index.html"
    text = path.read_text(encoding="utf-8")
    gallery_note = ('<p class="note">이번 페이지는 새 OOD v2의 정답(GT)입니다. '
                    'state_mlp·DeepSets·hierarchical·contact_mlp의 5 epoch 체크포인트를 '
                    '새 두 장면의 1,000개 전체에서 평가했습니다. '
                    '<a href="../../ood_comparison.html">새 OOD v2 모델 오차·조건별 비교 보기</a> · '
                    '<a href="../../ood_comparison_v1.html">기존 W&amp;B v1 기록</a></p>')
    text, count = re.subn(r'<p class="note">이번 페이지는 새 OOD v2의 정답\(GT\)입니다\..*?</p>', lambda _: gallery_note, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError("Could not find GT gallery evaluation status")
    path.write_text(text, encoding="utf-8")

    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace('새 조건의 모델 평가는 아직 수행하지 않았으며, 기존 W&B 막대 그래프는 이전 `floor_wall` v1의 512개 평가 기록입니다.',
                        '새 조건에서 5 epoch·seed 0의 `state_mlp`, `deepsets`, `hierarchical`을 동일한 1,000개 전체에서 평가했습니다. [OOD v2 모델 비교](docs/ood_comparison.html)에서 전체·벽·V자 바닥의 오차와 발산 진단을 확인할 수 있습니다. 기존 W&B 막대 그래프는 이전 `floor_wall` v1의 512개 평가 기록이며 [v1 보관 문서](docs/ood_comparison_v1.html)에 남겼습니다.')
    text = text.replace('위 평가 명령은 향후 실행 예시이며 v2 모델 평가 완료를 뜻하지 않습니다.',
                        'OOD v2는 5 epoch·seed 0의 `state_mlp`, `deepsets`, `hierarchical` 체크포인트를 같은 1,000개(벽 500개 + V자 바닥 500개)에서 100스텝 평가했습니다. 결과는 [OOD v2 비교 문서](docs/ood_comparison.html), 원본은 `outputs/ood-v2-evaluation/{model}/evaluation.json`에 저장했습니다. 기존 W&B v1 기록은 [별도 문서](docs/ood_comparison_v1.html)로 보존합니다.')
    text = text.replace('`state_mlp`, `deepsets`, `hierarchical`을 동일한 1,000개',
                        '`state_mlp`, `deepsets`, `hierarchical`, `contact_mlp`를 동일한 1,000개')
    text = text.replace('`state_mlp`, `deepsets`, `hierarchical` 체크포인트를 같은 1,000개',
                        '`state_mlp`, `deepsets`, `hierarchical`, `contact_mlp` 체크포인트를 같은 1,000개')
    text = text.replace('기존 3종 모델의 OOD 결과에 본 학습을 완료하지 않은 이 모델의 수치를 추가하지 않습니다.',
                        '5 epoch 학습을 완료한 `contact_mlp`를 기존 세 모델과 같은 OOD v2 1,000개·100스텝 조건에서 평가했습니다. 네 모델의 결과는 [OOD 비교 문서](docs/ood_comparison.html)에서 확인할 수 있습니다.')
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/assets/ood-v2-comparison/ood-data.json")
    parser.add_argument("--skip-related", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    assert data["protocol"]["evaluated_trajectories"] == 1000
    assert data["protocol"]["all_models_same_trajectory_ids"]
    assert data["protocol"]["scenario_counts"] == {"toss_wall": 500, "toss_valley": 500}
    assert data["protocol"]["horizon_steps"] == 100
    docs = ROOT / "docs"
    css = read_page_style(docs)
    files = {scope: {group: {kind: relative_artifact(path) for kind, path in entry.items()} for group, entry in groups.items()} for scope, groups in data["files"].items()}
    for scope in SCOPES:
        for group in GROUPS:
            for ext in ("png", "svg"):
                assert (docs / files[scope][group][ext]).is_file(), files[scope][group][ext]
    inline = dict(data, files=files)
    values = data["mean_errors"]["rollout"]
    base = values["state_mlp"]["position_rmse"]
    gains = {m: reduction_text((1 - values[m]["position_rmse"] / base) * 100) for m in MODELS[1:]}
    winner_note = "전체 100스텝 rollout에서 " + ", ".join(
        f"{label}는 {min(MODELS, key=lambda m: values[m][metric])}가 가장 낮습니다"
        for metric, label, _ in FIELDS
    ) + ". 조건별 결과는 아래 선택 메뉴에서 확인하세요."
    stats = "".join(f'<div class="stat"><strong>{values[m]["position_rmse"]:.5f} m</strong><span>{m} 위치 RMSE</span><small>{"요청 100스텝 · 유효 구간 평균" if m == "state_mlp" else "state_mlp 대비 " + gains[m]}</small></div>' for m in MODELS)
    group_buttons = "".join(f'<button type="button" data-group="{g}" aria-pressed="{str(g == "rollout").lower()}">{v[0]}</button>' for g, v in GROUPS.items())
    notes = "".join(f'<p class="small muted"><strong>{label}:</strong> {note}</p>' for label, note in GROUPS.values())
    source_rows = "".join(f'<tr><th scope="row">{m}</th><td class="path"><code>{esc(data["checkpoints"][m]["path"])}</code></td><td>{data["checkpoints"][m]["epoch"]}</td><td>{data["checkpoints"][m]["seed"]}</td><td><a href="../outputs/ood-v2-evaluation/{m}/evaluation.json">원본 JSON</a></td></tr>' for m in MODELS)
    page = Template(r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="OOD v2 벽 투척 500개와 각 평면 10도 V자 바닥 500개에서 contact_mlp, state_mlp, DeepSets, hierarchical의 실제 평가 결과"><title>OOD v2 비교 · Contact MLP와 세 모델</title><style>$css
.scope-select{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:16px 20px;background:#eaf3f7;border-radius:12px;margin:24px 0 12px}.scope-select select{min-width:235px}.scope-annotation{font-size:14px;color:#426375;margin:8px 0 18px}.diagnostic-table td small{color:#577087}.gain.bad{background:#fff4ed;border-color:#e8c9b2;border-top-color:#c07436}.gain.bad .large{color:#a55d25}.stat small{line-height:1.5}.chart-tabs{flex-wrap:wrap}.chart{background:white}.scope-grid video{aspect-ratio:4/3}.alert{border-left-color:#c58336;background:#fff5e8}.comparison-summary{font-size:15px;line-height:1.9}.compact{font-size:12px}.path code{overflow-wrap:anywhere;white-space:normal}.scope-grid .mini-card{padding:22px}.diagnostic-table th,.diagnostic-table td{padding:11px 10px}@media(max-width:650px){.scope-select{padding:14px}.scope-select select{min-width:0;width:100%}}
.stats{grid-template-columns:repeat(4,minmax(0,1fr))}.gain-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.gain.contact{background:#fff8ee;border-color:#ead2ad;border-top-color:#c57624}.gain.contact .large{color:#a85e18}.gain .large{font-size:34px}.model-note{background:#f4f7fb;border:1px solid #d4dfea;border-radius:12px;padding:18px 22px;margin:22px 0}.model-note p{margin:8px 0}.stat:last-child strong{color:#ffce91}@media(max-width:900px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.gain-grid{grid-template-columns:1fr}}@media(max-width:420px){.stats{grid-template-columns:1fr}}
</style></head><body><header class="top"><div class="top-inner"><a class="brand" href="deepsets_explained.html"><b>CUBE</b> / OOD v2</a><nav aria-label="페이지 이동"><a href="deepsets_explained.html">DeepSets 설명</a><a href="#comparison" aria-current="page">OOD 비교</a><a href="visualizations/ood-context/index.html">GT 갤러리</a><a href="nerd_goal.html">연구 목표</a><a href="ood_comparison_v1.html">이전 v1 기록</a></nav></div></header><main>
<section class="hero"><p class="eyebrow">OOD v2 · Contact MLP 추가 평가</p><h1>명시적 접촉 입력의 Contact MLP,<br>기존 세 모델과 같은 OOD에서 비교합니다.</h1><p class="intro">state_mlp, DeepSets, hierarchical에 contact_mlp를 추가했습니다. 모두 같은 10,000개 학습 데이터로 5 epoch·seed 0 학습한 체크포인트이며, OOD v2 1,000개 전체에서 100스텝을 예측합니다. 전체 평균과 조건별 결과, 접촉 진단을 함께 비교합니다.</p><div class="tags"><span class="tag">벽 투척 500개 · 50%</span><span class="tag">V자 바닥 500개 · 50%</span><span class="tag">각 평면 10° · 총 방향 변화 20°</span><span class="tag">100스텝 · 2초</span></div><div class="stats">$stats</div></section>
<p class="lead-note"><strong>출처:</strong> 기존 학습 체크포인트를 새 OOD v2에서 실행한 <strong>로컬 재평가</strong>입니다. 기존 W&amp;B의 floor_wall v1 수치는 <a href="ood_comparison_v1.html">별도 기록</a>으로 보존했습니다.</p>
<div class="model-note"><strong>Contact MLP의 입력과 비교 범위</strong><p>NeRD 방식으로 고정된 꼭짓점 8개에서 <strong>환경 표면점·법선·부호 있는 거리</strong>를 계산하고, 현재 속도·외력·중력 방향과 함께 71차원 입력으로 사용합니다. Point cloud 생성이나 pooling을 거치지 않으며, 자율 rollout의 접촉 정보는 매번 자신의 예측 상태에서 다시 계산합니다. 미래 정답 접촉이나 접촉력은 입력하지 않습니다.</p><p class="small muted">네 모델은 같은 상태 변화량 MSE + 회전각 제곱 손실과 같은 타깃 정규화를 사용합니다. Contact MLP의 입력 정규화는 학습 데이터만으로 계산했습니다. 입력 정보와 접촉 계산 방식이 다르므로, 결과는 전체 모델 구성의 비교입니다. 같은 꼭짓점에 여러 표면이 가까우면 부호 있는 거리가 가장 작은 표면 하나만 사용합니다.</p></div>
<p class="note comparison-summary"><strong>전체 지표별 결과:</strong> $winner_note</p>
<p class="note comparison-summary" id="scenario-finding"><strong>조건별 핵심 결과:</strong> $scenario_summary</p>
<section class="paper" id="comparison"><div class="section-head"><span class="section-no">01</span><h2>전체와 조건별 OOD 막대 그래프</h2></div><p class="subtitle">낮은 막대가 작은 오차를 뜻합니다. 같은 단위의 지표끼리 모델을 비교하며 모든 막대의 축은 0에서 시작합니다. 아래 수치는 trajectory마다 계산한 지표의 산술평균이며, 충돌·지속 접촉 지표는 해당 GT 사건이 있는 trajectory만 포함합니다.</p>
<div class="scope-select"><label for="scope-select"><strong>평가 데이터</strong></label><select id="scope-select"><option value="overall">전체 · 벽 500개 + V자 바닥 500개</option><option value="toss_wall">벽 투척 · 500개</option><option value="toss_valley">V자 바닥 · 500개</option></select></div><p id="scope-annotation" class="scope-annotation">전체 1,000개를 50:50으로 합쳐 평균한 결과이며, 두 조건의 차이는 선택 메뉴에서 따로 확인할 수 있습니다.</p>
<div class="chart-tabs" role="group" aria-label="평가 방식">$group_buttons</div><figure style="margin:0"><img id="ood-chart" class="chart" src="$first_chart" alt="OOD v2 전체 1,000개의 100스텝 rollout 오차 비교"><figcaption id="chart-caption" class="caption">평가 주석 — $rollout_note</figcaption></figure><details><summary>각 평가가 의미하는 것 · 한 문장 주석</summary>$notes</details><div class="asset-links"><a id="chart-svg" href="$first_svg" download>현재 그래프 SVG 저장</a><a href="assets/ood-v2-comparison/ood-data.json">지표·평가 조건·출처 JSON</a></div>
<div class="gain-grid"><div id="deep-card" class="gain"><h3>DeepSets · 위치 RMSE</h3><span class="large" id="deep-gain">$deep_gain</span><p id="deep-detail">state_mlp $base → DeepSets $deep m</p></div><div id="hier-card" class="gain purple"><h3>계층형 · hierarchical · 위치 RMSE</h3><span class="large" id="hier-gain">$hier_gain</span><p id="hier-detail">state_mlp $base → hierarchical $hier m</p></div><div id="contact-card" class="gain contact"><h3>Contact MLP · 위치 RMSE</h3><span class="large" id="contact-gain">$contact_gain</span><p id="contact-detail">state_mlp $base → contact_mlp $contact m</p></div></div>
<div class="table-wrap"><table><thead><tr><th>모델</th><th>위치 RMSE</th><th>회전 거리 평균</th><th>선속도 RMSE</th><th>각속도 RMSE</th></tr></thead><tbody id="metric-table">$metric_rows</tbody></table></div><p class="caption" id="metric-counts"></p><p class="note comparison-summary" id="comparison-summary">$summary</p>
<p class="caption">한 스텝 예측이 낮아도 장기 rollout이 안정적이라는 뜻은 아닙니다. 아래 발산율·100스텝 완주 수·오차 분포와 함께 해석해 주세요. 학습 seed는 하나이며, 통계적 유의성을 검정한 결과는 아닙니다.</p></section>
<section class="paper" id="diagnostics"><div class="section-head"><span class="section-no">02</span><h2>발산과 접촉 진단을 함께 보기</h2></div><p class="subtitle">위에서 선택한 전체·벽·V자 바닥 조건에 맞춰 갱신됩니다. 발산율은 평가기의 안전 종료 조건을 만족해 rollout이 중단된 trajectory의 비율입니다. <strong>조기 종료 사례의 오차는 종료 전 예측 프레임까지만 계산되므로, 짧게 끝난 rollout의 평균을 완주한 모델과 같은 의미로 읽으면 안 됩니다.</strong></p><div class="table-wrap"><table class="diagnostic-table"><thead><tr><th>모델</th><th>발산율</th><th>100스텝 완주</th><th>실제 rollout 스텝</th><th>평균 관통량</th><th>사례별 최대 관통량 평균</th><th>접촉 불일치율</th></tr></thead><tbody id="diagnostic-table">$diagnostics</tbody></table></div><p class="caption">관통량·접촉 불일치율은 예측된 궤적에서 계산한 진단값입니다. 평균 상태 오차와 별도로 물리적 접촉의 타당성을 확인합니다.</p><div class="note alert" id="stability-note"></div><h3>100스텝 rollout 위치 RMSE 분포</h3><p class="small muted">평균 외에 중앙값·95백분위·최댓값을 표시해 일부 큰 오차가 미치는 영향을 확인합니다. 단위는 m이며, 조기 종료 사례도 종료 전 유효 구간의 지표로 포함합니다.</p><div class="table-wrap"><table><thead><tr><th>모델</th><th>중앙값</th><th>95백분위</th><th>최댓값</th></tr></thead><tbody id="distribution-table">$distribution</tbody></table></div></section>
<section class="paper" id="scope"><div class="section-head"><span class="section-no">03</span><h2>두 OOD 장면이 확인하는 것</h2></div><div class="scope-grid"><article class="mini-card"><h3>50% · 벽을 향한 투척 + 평평한 바닥</h3><p class="small muted">랜덤 위치·회전에서 출발하고, 큐브가 바닥보다 먼저 벽에 도달하도록 벽 방향의 초기 속도를 정합니다. 바닥과 벽은 모두 평면입니다. <strong>벽의 형상 변화와 초기 x방향 속도의 분포 변화가 함께 포함되므로, 순수한 지형 효과만 분리한 실험은 아닙니다.</strong></p><p class="note small"><strong>평가 의미 · 한 문장:</strong> 처음 만나는 벽에 충돌한 후 평평한 바닥에 닿을 때, 학습과 다른 투척 속도와 접촉 전환에 따른 반사·회전을 예측하는지 확인합니다.</p><video controls playsinline preload="metadata" poster="visualizations/ood-context/toss_wall/trajectory_12000/gt.png"><source src="visualizations/ood-context/toss_wall/trajectory_12000/gt.mp4" type="video/mp4"></video><p class="caption">저장된 GT · toss_wall · ID 12000 · 2초</p><a href="visualizations/ood-context/toss_wall/index.html">벽 투척 GT 6개 보기 →</a></article><article class="mini-card"><h3>50% · 학습과 같은 투척 + V자 바닥</h3><p class="small muted">초기 위치·회전·선속도·각속도는 train과 같은 설정에서 뽑고 바닥을 두 평면으로 바꿉니다. <strong>각 평면이 수평에서 10°씩 기울고 중앙이 낮은 V자</strong>이며, 두 평면의 총 방향 변화는 20°입니다.</p><p class="note small"><strong>평가 의미 · 한 문장:</strong> 초기 투척 분포를 유지하면서 위치에 따라 법선이 달라지는 두 바닥 평면의 지지·접촉 전환에 맞춰 움직임을 예측하는지 확인합니다.</p><video controls playsinline preload="metadata" poster="visualizations/ood-context/toss_valley/trajectory_12500/gt.png"><source src="visualizations/ood-context/toss_valley/trajectory_12500/gt.mp4" type="video/mp4"></video><p class="caption">저장된 GT · toss_valley · ID 12500 · 2초</p><a href="visualizations/ood-context/toss_valley/index.html">V자 바닥 GT 6개 보기 →</a></article></div><p class="caption">영상은 실제 저장된 시뮬레이션 정답(GT)입니다. 모델 예측 영상이나 새로 생성한 설명 그림이 아닙니다.</p></section>
<section class="paper" id="goal"><div class="section-head"><span class="section-no">04</span><h2>다음 목표: 단순 경사보다 복잡한 구덩이·융기</h2></div><p><strong>하나의 평면 기울기 변화로 표현되지 않는 구덩이·융기, 그리고 경험하지 못한 장애물에도 반응하는 NeRD</strong>가 목표입니다. 현재 V자 바닥은 위치에 따라 표면 법선이 달라지는 두 평면 조건이며, 임의 구덩이·융기·단차·장애물 일반화는 다음 검증 항목입니다.</p><a class="link-button" href="nerd_goal.html">연구 목표와 후속 평가 보기 →</a></section>
<section class="paper" id="sources"><div class="section-head"><span class="section-no">05</span><h2>실제 평가 조건과 재현 근거</h2></div><p>학습 데이터 <code>cube-toss</code>의 10,000개 trajectory로 학습한 같은 5 epoch·seed 0의 <code>best.pt</code>를 사용했습니다. 새 학습 없이 <code>data/cube-toss-ood-v2</code>의 ID 12000–12999 전체를 평가했으며, 네 모델의 평가 ID와 100스텝 요청 길이는 같습니다. dt는 0.02초이고 외력·토크 action은 모든 시점에서 0입니다.</p><div class="table-wrap"><table><thead><tr><th>모델</th><th>사용한 체크포인트</th><th>epoch</th><th>seed</th><th>실제 평가 원본</th></tr></thead><tbody>$sources</tbody></table></div><div class="asset-links"><a href="assets/ood-v2-comparison/ood-data.json">모든 지표·데이터 fingerprint·SHA256</a><a href="../configs/cube_toss_ood_v2.json">OOD v2 생성 설정</a><a href="ood_comparison_v1.html">기존 W&amp;B v1 기록</a></div><p class="small muted">이번 표와 그래프의 출처는 새 OOD v2의 로컬 평가 JSON입니다. 기존 W&amp;B의 v1 run.summary를 v2 결과로 재사용하거나 덮어쓰지 않았습니다. 조건별 평균은 같은 원본의 scenario를 나누어 계산했으며, 95백분위는 선형 보간을 사용합니다.</p><details><summary>추론 방식과 수치 검증</summary><p class="small muted">자율 rollout은 모델별 메모리 사용량에 맞춰 같은 장면의 trajectory를 float32 배치로 묶어 추론하며, 각 trajectory의 예측 이력은 독립적으로 유지합니다. 실제 모델별 배치 크기는 원본 평가 JSON에 기록했습니다. 기존 평가 경로와 정답 상태를 입력하는 한 스텝 예측을 대조했습니다. 장기 자율 rollout은 부동소수점의 작은 차이가 접촉 이후 누적될 수 있으므로 기존 실행과 프레임별로 동일하다고 가정하지 않습니다. 원본 평가 JSON에 추론 조건과 수치 민감도 검증 근거를 남겼습니다. 지연 시간은 단일 trajectory의 공식 성능 측정값으로 사용하지 않습니다.</p></details></section><footer><span>큐브 접촉 동역학 / OOD v2 실제 평가</span><span>문서 생성 $timestamp (한국 시간)</span></footer></main><script type="application/json" id="ood-values">$data</script><script>
(()=>{'use strict';
const data=JSON.parse(document.getElementById('ood-values').textContent),el=id=>document.getElementById(id);
const models=data.models,fields=[['position_rmse','m'],['rotation_geodesic_rad','rad'],['linear_velocity_rmse','m/s'],['angular_velocity_rmse','rad/s']];
const groups=$groups,scopes=$scopes;
let group='rollout',scope='overall';
const num=(v,d=5)=>Number.isFinite(v)?v.toFixed(d):'—';
const change=(v,b)=>Number.isFinite(v)&&Number.isFinite(b)&&b!==0?(1-v/b)*100:null;
const changeText=p=>p===null?'비교 불가':Math.abs(p)<.05?'차이 0.0%':Math.abs(p).toFixed(1)+'% '+(p>0?'감소':'증가');
function render(){
const view=scope==='overall'?data:data.scenarios[scope],values=view.mean_errors[group],baseline=values.state_mlp;
document.querySelectorAll('[data-group]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.group===group)));
el('ood-chart').src=data.files[scope][group].png;el('ood-chart').alt='OOD v2 '+scopes[scope]+' '+groups[group][0]+' 상태 오차 비교';el('chart-svg').href=data.files[scope][group].svg;
el('chart-caption').textContent='평가 주석 — '+groups[group][1];
el('scope-annotation').textContent=scope==='overall'?'전체 1,000개를 50:50으로 합쳐 평균한 결과이며, 두 조건의 차이는 선택 메뉴에서 따로 확인할 수 있습니다.':scope==='toss_wall'?'벽을 향하도록 초기 x방향 속도를 바꾸었으므로, 벽의 추가와 초기 속도 분포 변화가 함께 포함됩니다.':'초기 투척 설정은 train과 같고, 바닥만 각 평면 10°인 중앙이 낮은 V자로 바꾼 조건입니다.';
el('metric-table').innerHTML=models.map(m=>'<tr><th scope="row">'+m+'</th>'+fields.map(([k,u])=>'<td><strong>'+num(values[m][k])+'</strong> '+u+'<small>'+(m==='state_mlp'?'기준 모델':'state_mlp 대비 '+changeText(change(values[m][k],baseline[k])))+'</small></td>').join('')+'</tr>').join('');
el('metric-counts').textContent='해당 평가에 포함된 trajectory: '+models.map(m=>m+' '+view.metric_trajectory_counts[m][group]+'개').join(' · ')+'.';
[['deepsets','deep'],['hierarchical','hier'],['contact_mlp','contact']].forEach(([m,id])=>{const p=change(values[m].position_rmse,baseline.position_rmse);el(id+'-gain').textContent=changeText(p);el(id+'-detail').textContent=groups[group][0]+' 위치 RMSE: '+num(baseline.position_rmse)+' → '+num(values[m].position_rmse)+' m';el(id+'-card').classList.toggle('bad',p!==null&&p<0);});
const counts=models.slice(1).map(m=>[m,fields.filter(([k])=>values[m][k]<baseline[k]).length]);
el('comparison-summary').textContent=scopes[scope]+'의 '+groups[group][0]+'에서 '+counts.map(([m,n])=>m+'은 네 가지 상태 지표 중 '+n+'개가 state_mlp보다 낮습니다').join(', ')+'. 지표별 차이는 위 표에 표시했습니다.';
el('diagnostic-table').innerHTML=models.map(m=>{const d=view.diagnostics[m],s=view.rollout_steps[m];return '<tr><th scope="row">'+m+'</th><td>'+num(view.divergence_rate[m]*100,1)+'%</td><td>'+s.completed_horizon_count+' / '+(s.completed_horizon_count+s.truncated_count)+'</td><td>'+s.min+'–'+s.max+'<small>평균 '+num(s.mean,1)+'</small></td><td>'+num(d.penetration_mean)+' m</td><td>'+num(d.penetration_max)+' m</td><td>'+num(d.contact_disagreement*100,2)+'%</td></tr>';}).join('');
const truncated=models.filter(m=>view.rollout_steps[m].truncated_count>0);
el('stability-note').textContent=truncated.length?'주의할 결과: '+truncated.map(m=>m+' '+view.rollout_steps[m].truncated_count+'개가 100스텝 전에 종료').join(', ')+'. 종료 전 유효 구간의 평균 오차와 전체 완주 성능을 구분해야 합니다.':'네 모델 모두 선택한 조건의 모든 trajectory에서 요청한 100스텝을 완주했습니다. 발산이 없다는 사실과 접촉이 물리적으로 정확하다는 것은 별도 기준입니다.';
el('distribution-table').innerHTML=models.map(m=>{const d=view.rollout_distribution?.[m]?.position_rmse||{};return '<tr><th scope="row">'+m+'</th><td>'+num(d.median)+'</td><td>'+num(d.p95)+'</td><td>'+num(d.max)+'</td></tr>';}).join('');
}
document.querySelectorAll('[data-group]').forEach(b=>b.addEventListener('click',()=>{group=b.dataset.group;render();}));el('scope-select').addEventListener('change',e=>{scope=e.target.value;render();});render();
})();</script></body></html>''').substitute(
        css=css, stats=stats, group_buttons=group_buttons, notes=notes,
        first_chart=files["overall"]["rollout"]["png"], first_svg=files["overall"]["rollout"]["svg"], rollout_note=GROUPS["rollout"][1],
        deep_gain=gains["deepsets"], hier_gain=gains["hierarchical"], contact_gain=gains["contact_mlp"], contact=f"{values['contact_mlp']['position_rmse']:.5f}", winner_note=winner_note, base=f"{base:.5f}", deep=f"{values['deepsets']['position_rmse']:.5f}", hier=f"{values['hierarchical']['position_rmse']:.5f}",
        metric_rows=table_rows(data), diagnostics=diagnostic_rows(data), distribution=distribution_rows(data), sources=source_rows,
        summary=summary_note(data), scenario_summary=scenario_note(data), groups=json.dumps(GROUPS, ensure_ascii=False), scopes=json.dumps(SCOPES, ensure_ascii=False),
        data=json.dumps(inline, ensure_ascii=False).replace("</", "<\\/"), timestamp=datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
    )
    (docs / "ood_comparison.html").write_text(page, encoding="utf-8")
    if not args.skip_related:
        update_related(docs, data)
    print(json.dumps({"page": "docs/ood_comparison.html", "scopes": list(SCOPES), "groups": list(GROUPS), "summary": summary_note(data)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
