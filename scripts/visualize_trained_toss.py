"""Render the three available toss models using the existing rollout gallery."""
from datetime import datetime
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from cube.visualization import GALLERY_STYLE, visualize_rollout


def main():
    models = ("state_mlp", "deepsets", "hierarchical")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-model", choices=models)
    args = parser.parse_args()
    out = Path("docs/visualizations/trained-toss-20261019")
    snapshots = Path("outputs/visualization-checkpoints/trained-toss-20261019")
    if args.refresh_model:
        report = json.loads((out / "checkpoints.json").read_text(encoding="utf-8"))
        assert report["complete"], "Previous gallery must be complete before refresh"
        report["updated_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        selected_models = (args.refresh_model,)
    else:
        out.mkdir(parents=True, exist_ok=False)
        snapshots.mkdir(parents=True, exist_ok=False)
        report = {"created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                  "dataset": "data/cube-toss", "split": "test", "scenario": "toss",
                  "horizon": 100, "limit": 6, "models": {}}
        selected_models = models
    report["complete"] = False
    # Freeze every input before rendering; training may continue independently.
    for model in selected_models:
        source = Path(f"outputs/cube-toss-suite-final/{model}_budget10000_seed0/best.pt")
        target = snapshots / f"{model}-best.pt"
        shutil.copy2(source, target)
        report["models"][model] = {"source_checkpoint": str(source),
                                   "snapshot_checkpoint": str(target),
                                   "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (out / "checkpoints.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifests = {}
    for model in models:
        if model in selected_models:
            result = visualize_rollout(
                report["dataset"], report["models"][model]["snapshot_checkpoint"],
                str(out / model), split="test", scenario="toss", limit=6, horizon=100,
                device="cpu", width=640, height=480, stride=1)
            print(json.dumps(result), flush=True)
        manifests[model] = json.loads((out / model / "manifest.json").read_text(encoding="utf-8"))
        report["models"][model]["checkpoint_meta"] = manifests[model]["trajectories"][0]["checkpoint_meta"]
    ids = manifests[models[0]]["selected_ids"]
    assert all(manifests[m]["selected_ids"] == ids and manifests[m]["complete"] for m in models)
    report["selected_ids"] = ids
    cards, rows = [], []
    for model in models:
        meta = report["models"][model]["checkpoint_meta"]
        # Runtime stores the completed epoch count (or active epoch for partial saves).
        epoch = meta.get("epoch")
        epoch_label = str(epoch) if epoch is not None else "unknown"
        rows.append(f'<tr><td><a href="{model}/index.html">{model}</a></td><td>{epoch_label}</td>'
                    f'<td>{meta.get("global_step", "unknown")}</td><td>{meta.get("val_loss", "unknown")}</td></tr>')
    for index, identity in enumerate(ids):
        panels = []
        for model in models:
            row = manifests[model]["trajectories"][index]
            prefix = f'{model}/{row["directory"]}'
            metrics = row["rollout_metrics"]
            panels.append(f'''<section><h3>{model}</h3><p>{row['completed_steps']} / 100 steps · position RMSE {metrics['position_rmse']:.4f} m · {'EARLY STOP' if row['diverged'] else 'ROLLOUT COMPLETE'}</p>
<video controls playsinline preload="metadata" poster="{prefix}/comparison.png"><source src="{prefix}/comparison.mp4" type="video/mp4"></video>
<p><a href="{prefix}/comparison.mp4" download>MP4 저장</a> · <a href="{prefix}/metrics.json">지표</a></p>
<details><summary>상태 변화와 오차 그래프</summary><img src="{prefix}/timeseries.png" loading="lazy" alt="{model} trajectory {identity} error"></details></section>''')
        cards.append(f'<article><h2>Trajectory {identity:05d}</h2><p>각 영상의 왼쪽: GT · 오른쪽: 모델 자율 rollout</p><button class="play">세 모델 함께 처음부터 재생</button><div class="panels">{"".join(panels)}</div></article>')
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>학습된 세 모델 · 큐브 토스 rollout</title><style>{GALLERY_STYLE}
.panels{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ccdce2}}button{{padding:10px 16px;cursor:pointer}}section p{{font-size:13px}}@media(max-width:1100px){{.panels{{grid-template-columns:1fr}}}}</style></head><body><main>
<header><div class="eyebrow">CUBE LAB / TRAINED TOSS CHECKPOINTS</div><h1>학습된 세 모델의 큐브 토스 rollout</h1><p>state_mlp · deepsets · hierarchical / seed 0 / 학습 예산 10,000 trajectories</p><p>생성: {html.escape(report['created_at'])} · 갱신: {html.escape(report.get('updated_at', report['created_at']))}</p></header>
<article><h2>비교 조건과 체크포인트</h2><p>data/cube-toss의 test/toss ID {', '.join(map(str, ids))}. 동일한 초기 상태에서 100 transitions(2초)를 자율 예측합니다. 모든 action은 0이며 미래 GT로 예측을 교체하지 않습니다. 기존 시각화 구현을 그대로 사용했습니다. 영상은 1280×480, 50 FPS입니다.</p>
<p>실행 시작 시 각 모델의 validation 기준 best.pt를 복사해 고정했습니다. 아래 epoch은 선택된 체크포인트가 저장된 학습 epoch이며 모델별 전체 학습 완료를 의미하지 않습니다.</p>
<table><thead><tr><th>모델 / 상세 갤러리</th><th>선택된 epoch (1부터)</th><th>Optimizer step</th><th>Validation loss</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><a href="checkpoints.json">체크포인트 원본 경로·SHA256·메타데이터</a></p><p>GT와 예측은 같은 카메라를 사용합니다. 선택된 6개 예시의 오차이며 전체 테스트 성능을 대표하지 않습니다. ROLLOUT COMPLETE는 수치 계산 완료를 뜻합니다.</p>
<label>재생 속도 <select id="speed"><option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option><option value="1">1×</option></select></label></article>
{''.join(cards)}</main><script>
const speed=document.getElementById('speed');function rate(){{document.querySelectorAll('video').forEach(v=>v.playbackRate=Number(speed.value));}}speed.addEventListener('change',rate);rate();
document.querySelectorAll('.play').forEach(b=>b.addEventListener('click',async()=>{{const videos=[...b.closest('article').querySelectorAll('video')];videos.forEach(v=>{{v.pause();v.currentTime=0;v.playbackRate=Number(speed.value);}});await Promise.all(videos.map(v=>v.play()));}}));
</script></body></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    report["complete"] = True
    (out / "checkpoints.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
