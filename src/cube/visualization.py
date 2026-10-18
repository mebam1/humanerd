"""Dataset and learned-rollout galleries with state-faithful video and diagnostics."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from cube.experiments.common import load_trajectory, state_metrics, write_json
from cube.experiments.learning import load_checkpoint
from cube.experiments.tracking import progress
from cube.experiments.training_runtime import resolve_device
from cube.rendering import render_states
from cube.visualization_data import predict_trajectory, select_trajectories


def _validate_options(width, height, stride, horizon):
    for key, value in (("width", width), ("height", height), ("stride", stride)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if width % 2 or height % 2:
        raise ValueError("Video width and height must be even for H.264 encoding")
    if horizon is not None and (isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1):
        raise ValueError("horizon must be a positive number of transitions")


def _plot_states(path, states, actions, dt, *, predicted=None, contacts=None, penetration=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.arange(len(states))*dt
    nq = (states.shape[-1]+1)//2
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    colors = ["#167c88", "#c75d39", "#6253a7"]
    for column, name, color in zip(range(3), ("x", "y", "z"), colors):
        axes[0, 0].plot(times, states[:, column], color=color, label=f"GT {name}")
        axes[0, 1].plot(times, states[:, nq+column], color=color, label=f"GT {name}")
        if predicted is not None:
            axes[0, 0].plot(times, predicted[:, column], "--", color=color, label=f"Model {name}")
            axes[0, 1].plot(times, predicted[:, nq+column], "--", color=color, label=f"Model {name}")
    axes[0, 0].set(title="Root position" + (" (solid: GT, dashed: model)" if predicted is not None else " (GT)"), ylabel="m")
    axes[0, 1].set(title="Root linear velocity in world coordinates", ylabel="m/s")
    if predicted is not None:
        error = np.linalg.norm(predicted[:, :3]-states[:, :3], axis=1)
        qp, qt = predicted[:, 3:7], states[:, 3:7]
        dot = np.abs((qp*qt).sum(-1)) / np.maximum(np.linalg.norm(qp, axis=-1)*np.linalg.norm(qt, axis=-1), 1e-12)
        rotation = np.rad2deg(2*np.arccos(np.clip(dot, 0, 1)))
        axes[1, 0].plot(times, error, color="#c75d39", label="Euclidean position error")
        axes[1, 0].set(title="Autonomous rollout: position error", ylabel="m")
        axes[1, 1].plot(times, rotation, color="#6253a7", label="Quaternion geodesic angle")
        axes[1, 1].set(title="Autonomous rollout: rotation error", ylabel="degrees")
    else:
        if np.count_nonzero(actions) == 0:
            for column, name, color in zip(range(3), ("x", "y", "z"), colors):
                axes[1, 0].plot(times, states[:, nq+3+column], color=color, label=f"omega {name}")
            axes[1, 0].set(title="Root angular velocity (local frame); all actions = 0", ylabel="rad/s")
        else:
            for channel in range(actions.shape[-1]):
                axes[1, 0].step(times[:-1], actions[:, channel], where="post", label=f"u{channel}")
            axes[1, 0].set(title="Recorded actions (world wrench / joint control)", ylabel="action channel value")
        if contacts is not None:
            axes[1, 1].step(times, contacts[:len(states)], where="post", color="#167c88", label="Contact count")
            axes[1, 1].set(title="Stored simulator contact diagnostics", ylabel="count")
            if penetration is not None:
                second = axes[1, 1].twinx()
                second.plot(times, penetration[:len(states)], color="#c75d39", label="Penetration (m)")
                second.set_ylabel("penetration (m)", color="#c75d39")
        else:
            for column, name in enumerate(("w", "x", "y", "z")):
                axes[1, 1].plot(times, states[:, 3+column], label=f"q{name}")
            axes[1, 1].set(title="Root quaternion (wxyz)", ylabel="component")
    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=.2)
        axis.legend(loc="best", fontsize=8, ncol=2)
    try:
        figure.savefig(path, dpi=140)
    finally:
        plt.close(figure)


GALLERY_STYLE = """
:root{font-family:system-ui,'Segoe UI','Malgun Gothic',sans-serif;color:#173544;background:#f2f6f8;line-height:1.7}
*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:32px 24px 60px}
header{padding:30px;background:#143d4d;color:#f0f8fa;border-radius:14px}header p{color:#c3dfe7}
h1{font-size:30px;line-height:1.35;margin:8px 0 14px}h2{font-size:22px;margin:0}a{color:#067e8c}
header a{color:#8be2e7}.eyebrow{font-size:12px;letter-spacing:2px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:24px 0}
select{padding:7px 12px;border:1px solid #a6bfc9;border-radius:6px;background:white;font:inherit}
article{background:white;border:1px solid #d5e2e7;border-radius:12px;padding:24px;margin:24px 0;box-shadow:0 4px 18px #153d4d08}
.meta{color:#526e7b;font-size:14px}.badge{padding:4px 10px;border-radius:15px;background:#e4f4ed;color:#216b4d;font-size:12px}
.bad{background:#fff0e8;color:#9d3819}.status{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
video{display:block;width:100%;background:#13222c;border-radius:8px;margin:18px 0}img{width:100%;height:auto;border-radius:8px}
.links{display:flex;gap:16px;flex-wrap:wrap;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#eff5f7;border-radius:8px;font-size:12px}
.note{border-left:4px solid #df9a3d;padding:10px 16px;background:#fff8ed}.metrics{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0}.metrics b{display:block;font-size:20px}
details{margin-top:18px}summary{cursor:pointer}footer{color:#526e7b;font-size:12px;margin-top:28px}
@media(max-width:600px){main{padding:16px 10px}article,header{padding:18px}h1{font-size:24px}}
"""


def _write_gallery(output, report):
    escape = html.escape
    model_mode = report["mode"] == "model_rollout"
    title = "GT · Model rollout 비교" if model_mode else "학습 데이터 · Ground truth"
    intro = ("왼쪽은 저장된 GT, 오른쪽은 모델의 자율 예측입니다. 두 화면은 같은 시각과 카메라를 사용합니다. "
             "모델은 초기 GT 상태와 기록된 action만 받아 예측 상태를 다음 입력으로 사용합니다."
             if model_mode else "저장된 시뮬레이터 상태를 그대로 재생합니다. 새로운 물리 trajectory를 생성하지 않습니다.")
    cards = []
    for row in report["trajectories"]:
        prefix = row["directory"]
        video = f"{prefix}/{row['video_name']}"
        poster = f"{prefix}/{Path(row['video_name']).stem}.png"
        status = "EARLY STOP" if row.get("diverged") else "RECORDED GT" if not model_mode else "ROLLOUT COMPLETE"
        metrics = row.get("rollout_metrics")
        metric_html = ""
        if metrics:
            metric_html = (f'<div class="metrics"><span>Root position RMSE<b>{metrics["position_rmse"]:.4f} m</b></span>'
                           f'<span>Rotation geodesic mean<b>{np.rad2deg(metrics["rotation_geodesic_rad"]):.2f}°</b></span>'
                           f'<span>Linear velocity RMSE<b>{metrics["linear_velocity_rmse"]:.4f} m/s</b></span></div>')
        warning = f'<p class="note">예측이 중단됐습니다: {escape(str(row["reason"]))}. 유효한 예측 구간까지만 재생합니다.</p>' if row.get("diverged") else ""
        initial_html = ""
        if "initial_condition" in row:
            initial = row["initial_condition"]
            vector = lambda values: "[" + ", ".join(f"{value:.3f}" for value in values) + "]"
            action_summary = row["action_summary"]
            action_label = "모든 action = 0" if action_summary["all_zero"] else f'max |action| = {action_summary["max_abs"]:.4f}'
            initial_html = (f'<p class="meta">초기 위치 {vector(initial["position_world"])} m · '
                            f'회전 wxyz {vector(initial["quaternion_wxyz"])}</p>'
                            f'<p class="meta">초기 선속도 {vector(initial["linear_velocity_world"])} m/s · '
                            f'각속도(local) {vector(initial["angular_velocity_local"])} rad/s · {action_label}</p>')
        cards.append(f'''<article><div class="status"><h2>Trajectory {row['id']:05d}</h2><span class="badge {'bad' if row.get('diverged') else ''}">{status}</span></div>
<p class="meta">{escape(row['split'])} / {escape(row['scenario'])} · seed {row['seed']} · {row['completed_steps']} transitions · {row['duration_seconds']:.2f} s</p>
{initial_html}{warning}{metric_html}<video controls preload="metadata" playsinline poster="{poster}"><source src="{video}" type="video/mp4"></video>
<div class="links"><a href="{video}" download>MP4</a><a href="{prefix}/trajectory.npz" download>Trajectory NPZ</a><a href="{prefix}/scene.json">Scene JSON</a><a href="{prefix}/metrics.json">Metrics JSON</a></div>
<details open><summary>상태 변화와 오차</summary><img src="{prefix}/timeseries.png" loading="lazy" alt="Trajectory {row['id']} state time series"></details>
<details><summary>선택한 데이터와 모델 정보</summary><pre>{escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details></article>''')
    checkpoint_line = f'<p>Model: {escape(str(report.get("model_name")))} · checkpoint: {escape(str(report.get("checkpoint")))}</p>' if model_mode else ""
    completion_note = ('<p class="note">ROLLOUT COMPLETE는 선택한 구간의 수치 계산이 끝났다는 뜻입니다. '
                       '물리적으로 정확하다는 판정은 아닙니다. 예측이 바닥을 관통하거나 GT 기준 카메라 밖으로 '
                       '벗어나도 보정하지 않으며, 아래 상태 그래프와 오차에 그대로 기록합니다.</p>' if model_mode else '')
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · CUBE LAB</title><style>{GALLERY_STYLE}</style></head><body><main>
<header><div class="eyebrow">CUBE LAB / TRAJECTORY VIEWER</div><h1>{title}</h1><p>{intro}</p><p>Dataset: {escape(str(report['dataset']))} · source physics: {escape(report['backend'])} · {len(report['trajectories'])} trajectories</p>{checkpoint_line}</header>
<div class="toolbar"><label for="speed">영상 재생 속도</label><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><a href="manifest.json">전체 메타데이터</a></div>
<p class="note">{('기록된 action을 그대로 재생하는 open-loop 비교입니다. MPC나 새 action을 만드는 controller가 아닙니다. 선택한 trajectory의 예시이며 전체 성능 평가는 evaluate 결과로 판단하세요.' if model_mode else '기본 선택은 ID 오름차순이며 전체 데이터의 통계적 대표 표본은 아닙니다. split·scenario·trajectory ID를 바꿔 확인할 수 있습니다.')} 물리 시간 간격은 그대로 유지하고 stride는 영상 frame 수만 줄입니다.</p>
{completion_note}{''.join(cards)}<footer>Renderer: MuJoCo · Source data is read-only · Videos and diagnostics share the same exported states.</footer></main>
<script>document.getElementById('speed').addEventListener('change',function(){{document.querySelectorAll('video').forEach(v=>v.playbackRate=Number(this.value));}});</script></body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def _visualize(dataset, output, *, checkpoint=None, split=None, scenario=None, trajectory_id=None,
               limit=4, horizon=None, device="cpu", width=640, height=480, stride=1):
    _validate_options(width, height, stride, horizon)
    root, manifest, entries = select_trajectories(dataset, split, scenario, trajectory_id, limit,
                                                  default_split="test" if checkpoint else "train")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model_mode = checkpoint is not None
    loaded_checkpoint = load_checkpoint(checkpoint, str(resolve_device(device))) if model_mode else None
    report = {"mode": "model_rollout" if model_mode else "ground_truth", "dataset": str(dataset),
              "backend": manifest.get("backend", "unknown"), "kind": manifest.get("kind", "unknown"),
              "dataset_fingerprint": manifest.get("dataset_fingerprint"), "checkpoint": checkpoint,
              "model_name": loaded_checkpoint[1]["model_config"].get("name") if model_mode else None,
              "device": device if model_mode else None, "selected_ids": [entry["id"] for entry in entries],
              "camera": "fixed shared camera fitted to GT geometry", "stride": stride,
              "filters": {"split": split if split is not None else (None if trajectory_id is not None else "test" if model_mode else "train"),
                          "scenario": scenario, "trajectory_id": trajectory_id, "limit": limit, "horizon": horizon},
              "selection": "Deterministic ID order; illustrative selected trajectories, not an aggregate benchmark",
              "complete": False, "trajectories": []}
    write_json(out / "manifest.json", report)
    for entry in progress(entries, desc="Visualize model" if model_mode else "Visualize GT", unit="trajectory"):
        trajectory = load_trajectory(root, entry)
        truth, actions = np.asarray(trajectory["states"]), np.asarray(trajectory["actions"])
        available = len(actions)
        steps = min(horizon, available) if horizon is not None else available
        target = out / f"trajectory_{entry['id']:05d}"
        target.mkdir(parents=True, exist_ok=True)
        scene = entry["scene_spec"]
        dt = float(scene.get("dt", .02))
        row = {key: entry[key] for key in ("id", "split", "scenario", "seed")}
        row.update(directory=target.name, available_steps=available, requested_horizon=horizon,
                   completed_steps=steps, dt=dt, reason=None, diverged=False)
        if model_mode:
            predicted = predict_trajectory(checkpoint, scene, truth, actions,
                                            horizon=steps, device=device, loaded_checkpoint=loaded_checkpoint)
            row.update({key: value for key, value in predicted.items() if key not in {"predicted", "truth", "actions"}})
            row["requested_horizon"] = horizon
            truth, actions, prediction = predicted["truth"], predicted["actions"], predicted["predicted"]
            row["rollout_metrics"] = state_metrics(prediction[1:], truth[1:]) if len(prediction) > 1 else None
            np.savez_compressed(target / "trajectory.npz", truth=truth, predicted=prediction, actions=actions,
                                times=np.arange(len(truth))*dt)
            video_name = "comparison.mp4"
        else:
            truth, actions, prediction = truth[:steps+1], actions[:steps], None
            arrays = {"states": truth, "actions": actions, "times": np.arange(len(truth))*dt}
            for key in ("contact_count", "penetration", "min_distance"):
                if key in trajectory:
                    arrays[key] = trajectory[key][:steps+1]
            for key in ("impact_event", "persistent_contact", "substep_penetration", "first_impact_time"):
                if key in trajectory:
                    arrays[key] = trajectory[key][:steps]
            np.savez_compressed(target / "trajectory.npz", **arrays)
            video_name = "gt.mp4"
        nq = (truth.shape[-1]+1)//2
        row["initial_condition"] = {"position_world": truth[0, :3].tolist(),
                                    "quaternion_wxyz": truth[0, 3:7].tolist(),
                                    "linear_velocity_world": truth[0, nq:nq+3].tolist(),
                                    "angular_velocity_local": truth[0, nq+3:nq+6].tolist()}
        row["action_summary"] = {"all_zero": bool(np.count_nonzero(actions) == 0),
                                 "max_abs": float(np.max(np.abs(actions))) if actions.size else 0.0}
        write_json(target / "scene.json", scene)
        row["duration_seconds"] = row["completed_steps"]*dt
        row["video_name"] = video_name
        row["render"] = render_states(truth, scene, str(target / video_name), predicted=prediction,
                                     width=width, height=height, stride=stride, camera="fixed")
        _plot_states(target / "timeseries.png", truth, actions, dt, predicted=prediction,
                     contacts=trajectory.get("contact_count"), penetration=trajectory.get("penetration"))
        write_json(target / "metrics.json", row)
        report["trajectories"].append(row)
        write_json(out / "manifest.json", report)
    report["complete"] = True
    write_json(out / "manifest.json", report)
    _write_gallery(out, report)
    result = {"gallery": str(out / "index.html"), "manifest": str(out / "manifest.json"),
              "trajectories": len(entries), "selected_ids": report["selected_ids"], "mode": report["mode"]}
    try:
        result["localhost_path"] = "/" + (out / "index.html").resolve().relative_to((Path.cwd() / "docs").resolve()).as_posix()
    except ValueError:
        pass
    return result


def visualize_data(dataset, output, split=None, scenario=None, trajectory_id=None, limit=4,
                   horizon=None, width=640, height=480, stride=1):
    return _visualize(dataset, output, split=split, scenario=scenario, trajectory_id=trajectory_id,
                      limit=limit, horizon=horizon, width=width, height=height, stride=stride)


def visualize_rollout(dataset, checkpoint, output, split=None, scenario=None, trajectory_id=None,
                      limit=4, horizon=100, device="cpu", width=640, height=480, stride=1):
    return _visualize(dataset, output, checkpoint=checkpoint, split=split, scenario=scenario,
                      trajectory_id=trajectory_id, limit=limit, horizon=horizon, device=device,
                      width=width, height=height, stride=stride)
