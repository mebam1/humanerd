"""Render the balanced OOD v2 GT gallery while preserving the v1 record."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from cube.experiments.common import load_manifest
from cube.visualization import GALLERY_STYLE, visualize_data


def model_evaluation_note(dataset_fingerprint):
    """Only announce a finished benchmark when its verified dataset matches."""
    summary_path = Path(__file__).resolve().parents[1] / 'docs/assets/ood-v2-comparison/ood-data.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if (summary.get('dataset_identity', {}).get('fingerprint') == dataset_fingerprint
                and summary.get('protocol', {}).get('evaluated_trajectories') == 1000
                and summary.get('protocol', {}).get('all_models_same_trajectory_ids')):
            return ('이번 페이지는 새 OOD v2의 정답(GT)입니다. '
                    'state_mlp·DeepSets·hierarchical의 같은 5 epoch 체크포인트를 '
                    '새 두 장면의 1,000개 전체에서 평가했습니다. '
                    '<a href="../../ood_comparison.html">새 OOD v2 모델 오차·조건별 비교 보기</a> · '
                    '<a href="../../ood_comparison_v1.html">기존 W&amp;B v1 기록</a>')
    return ('이번 페이지는 새 OOD v2의 정답(GT)입니다. 기존 W&amp;B 막대 그래프는 '
            '이전 floor_wall OOD v1에서 측정한 모델 평가이며, '
            '새 두 장면의 모델 평가 수치는 아직 계산하지 않았습니다.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/cube-toss-ood-v2')
    parser.add_argument('--output', default='docs/visualizations/ood-context')
    parser.add_argument('--limit-per-scene', type=int, default=6)
    args = parser.parse_args()
    if args.limit_per_scene < 1:
        parser.error('--limit-per-scene must be positive')
    out = Path(args.output)
    legacy = out.with_name(out.name + '-v1')
    if (out / 'manifest.json').exists() and not legacy.exists():
        previous = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        if previous.get('dataset') != args.dataset:
            shutil.copytree(out, legacy)
    out.mkdir(parents=True, exist_ok=True)
    _, dataset = load_manifest(args.dataset)
    scenarios = ('toss_wall', 'toss_valley')
    counts = {s: sum(e['scenario'] == s and e['split'] == 'ood' for e in dataset['trajectories']) for s in scenarios}
    assert counts['toss_wall'] == counts['toss_valley'] == 500, counts
    labels = {'toss_wall': '벽을 향한 랜덤 투척 + 평평한 바닥',
              'toss_valley': 'train과 같은 랜덤 투척 + 각 10° V자 바닥'}
    notes = {'toss_wall': '랜덤 위치·회전에서 평평한 벽을 향해 날아가 충돌한 뒤 바닥에 내려오는 반응을 확인합니다.',
             'toss_valley': '학습과 같은 초기 투척 분포에서 두 경사면이 만나는 낮은 중앙과 새로운 접촉 조합에 대한 반응을 확인합니다.'}
    reports, rows, checks = {}, [], []
    sections = []
    preview = Image.new('RGB', (1440, 390 * ((2 * args.limit_per_scene + 2) // 3)), '#eef3f7')
    draw = ImageDraw.Draw(preview)
    for scenario in scenarios:
        target = out / scenario
        visualize_data(args.dataset, str(target), split='ood', scenario=scenario,
                       limit=args.limit_per_scene, horizon=100, width=960, height=720)
        report = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        assert report['complete']
        reports[scenario] = report
        cards = []
        for row in report['trajectories']:
            prefix = f'{scenario}/{row["directory"]}'
            count, minimum = 0, 960 * 720
            with imageio.get_reader(out / prefix / 'gt.mp4') as reader:
                for index, raw in enumerate(reader.iter_data()):
                    frame = np.asarray(raw)
                    pixels = frame[65:-35]
                    blue = ((pixels[..., 2] > pixels[..., 0] * 1.5)
                            & (pixels[..., 2] > pixels[..., 1] * 1.1) & (pixels[..., 1] > 35))
                    minimum = min(minimum, int(blue.sum()))
                    if int(blue.sum()) < 150:
                        raise RuntimeError(f'Cube not clearly visible: {scenario} / {row["id"]} / frame {index}')
                    if index == 25:
                        position = len(checks)
                        x, y = (position % 3) * 480, (position // 3) * 390
                        preview.paste(Image.fromarray(frame).resize((480, 360)), (x, y + 30))
                        draw.text((x + 8, y + 8), f'{scenario} / ID {row["id"]} / t=0.50s', fill='#142b42')
                    count += 1
            assert count == 101
            check = {'id': row['id'], 'scenario': scenario, 'frames_checked': count,
                     'minimum_visible_cube_pixels': minimum, 'width': 960, 'height': 720}
            checks.append(check)
            merged = deepcopy(row)
            merged['directory'] = prefix
            rows.append(merged)
            p = row['initial_condition']['position_world']
            v = row['initial_condition']['linear_velocity_world']
            vector = lambda a: '[' + ', '.join(f'{x:.2f}' for x in a) + ']'
            cards.append(f'''<article><h3>Trajectory {row['id']}</h3>
<p class="meta">초기 위치 {vector(p)} m · 초기 선속도 {vector(v)} m/s · 랜덤 초기 회전</p>
<video controls playsinline preload="metadata" poster="{prefix}/gt.png"><source src="{prefix}/gt.mp4" type="video/mp4"></video>
<div class="links"><a href="{prefix}/gt.mp4" download>MP4 저장</a><a href="{prefix}/metrics.json">메타데이터</a><a href="{prefix}/trajectory.npz" download>GT 상태</a></div>
<details><summary>상태·접촉 시계열</summary><img src="{prefix}/timeseries.png" loading="lazy" alt="{scenario} ID {row['id']}의 위치·속도·접촉 시계열"></details></article>''')
            print(json.dumps(check), flush=True)
        sections.append(f'''<section id="{scenario}" class="scenario"><div class="scene-title"><h2>{labels[scenario]}</h2><span class="badge">전체 OOD의 50% · 500개</span></div>
<p class="note">평가 의미 — {notes[scenario]}</p><p>조건별 첫 {len(cards)}개 ID를 골라 같은 100스텝(2초)의 저장된 GT를 재생합니다. <a href="{scenario}/index.html">이 조건의 상세 갤러리</a></p><div class="cards">{''.join(cards)}</div></section>''')
    created = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    report = {'version': 'OOD v2', 'mode': 'ground_truth', 'dataset': args.dataset,
              'dataset_fingerprint': dataset['dataset_fingerprint'], 'backend': dataset['backend'],
              'kind': 'cube', 'created_at': created, 'complete': True, 'dataset_counts': counts,
              'valley_angle_degrees_per_plane': 10, 'selected_ids': [r['id'] for r in rows],
              'selection': 'First IDs per scenario; illustrative GT only, not a model benchmark',
              'trajectories': rows}
    (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'render-verification.json').write_text(json.dumps({'frames_checked': sum(c['frames_checked'] for c in checks),
                                                            'videos': checks}, indent=2), encoding='utf-8')
    preview.save(out / 'ood-v2-preview.png')
    legacy_link = f'<a href="../{legacy.name}/index.html">이전 OOD v1 GT 기록</a>' if legacy.exists() else ''
    evaluation_note = model_evaluation_note(dataset['dataset_fingerprint'])
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OOD v2 · 랜덤 투척 GT</title><style>{GALLERY_STYLE}
.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.cards article{{min-width:0;margin:0;padding:20px}}.cards h3{{margin:0;font-size:19px}}.scenario{{margin-top:36px}}.scene-title{{display:flex;align-items:center;gap:15px;flex-wrap:wrap}}.scene-title h2{{font-size:24px}}.tabs{{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}}.tabs a{{padding:9px 14px;background:white;border:1px solid #bcd7d9;border-radius:9px;text-decoration:none}}header h1{{font-size:34px}}[hidden]{{display:none!important}}@media(max-width:850px){{.cards{{grid-template-columns:1fr}}header h1{{font-size:27px}}}}
</style></head><body><main><header><div class="eyebrow">CUBE LAB / OOD V2 / GROUND TRUTH</div><h1>랜덤 투척으로 바꾼 OOD 평가 데이터</h1><p>벽을 향한 투척 500개(50%) + 각 평면 10°인 V자 바닥 투척 500개(50%). 총 1,000개 OOD 중 각 조건 {args.limit_per_scene}개씩, {len(rows)}개 GT 영상을 보여 줍니다.</p>
<p>두 조건 모두 랜덤 위치·회전으로 시작하며 모든 action은 0입니다. V자 바닥은 train과 같은 초기 선속도·각속도 분포를 사용하고, 벽 조건은 바닥에 닿기 전 벽을 맞도록 x방향 초기 속도를 정합니다.</p><p>V자 바닥: 중앙이 낮고 양쪽이 높은 z = |x| tan(10°), 각 평면의 수평 대비 경사 10°.</p></header>
<div class="tabs"><a href="#toss_wall">① 벽 + 평평한 바닥 · 50%</a><a href="#toss_valley">② 각 10° V자 바닥 · 50%</a><a href="../../nerd_goal.html">연구 목표</a><a href="manifest.json">선택·데이터 메타데이터</a>{legacy_link}</div>
<div class="toolbar"><label for="speed">재생 속도</label><select id="speed"><option value="0.25">0.25배</option><option value="0.5" selected>0.5배</option><option value="1">1배</option><option value="2">2배</option></select><span class="meta">960×720 · 50 FPS · 각 2초 · 저장된 상태 재생</span></div>
<p class="note">{evaluation_note}</p>
{''.join(sections)}<footer>생성: {html.escape(created)} · 원본 데이터: {html.escape(args.dataset)} · 궤적마다 GT 기준 고정 카메라를 사용합니다.</footer></main><script>const speed=document.getElementById('speed');function rate(){{document.querySelectorAll('video').forEach(v=>v.playbackRate=Number(speed.value));}}speed.addEventListener('change',rate);rate();</script></body></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps({'gallery': str(out / 'index.html'), 'dataset_counts': counts, 'videos': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
