"""Evaluate fixed checkpoints on all OOD v2 episodes with same-scene FP32 batches.

Metrics use the repository's per-trajectory definitions. MuJoCo only inspects
predicted states after inference; it never corrects or integrates predictions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from cube.experiments.common import load_manifest, load_trajectory, model_state, state_metrics, write_json
from cube.experiments.learning import load_checkpoint
from cube.experiments.training_runtime import SceneCache, scene_key
from cube.models import apply_delta
from cube.visualization_data import predict_trajectory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer(model, truth, actions, device, transition_batch_size):
    """A member's inputs never depend on another trajectory or future GT."""
    if truth.shape[1:] != (101, 13) or actions.shape[1:] != (100, 6):
        raise ValueError('OOD v2 requires 100 Cube transitions and 101 states')
    states = torch.as_tensor(truth, dtype=torch.float32, device=device)
    control = torch.as_tensor(actions, dtype=torch.float32, device=device)
    count, horizon = control.shape[:2]
    flat_states = states[:, :-1].reshape(-1, states.shape[-1])
    flat_actions = control.reshape(-1, control.shape[-1])
    single = []
    for start in range(0, len(flat_states), transition_batch_size):
        x, u = flat_states[start:start + transition_batch_size], flat_actions[start:start + transition_batch_size]
        single.append(apply_delta(x, model_state(model, x, u)))
    one_step = torch.cat(single).reshape(count, horizon, -1).cpu().numpy()
    if not np.isfinite(one_step).all():
        raise RuntimeError('Non-finite one-step predictions; evaluation cannot report finite errors')
    predictions = torch.empty_like(states)
    predictions[:, 0] = states[:, 0]
    active = torch.ones(count, dtype=torch.bool, device=device)
    completed = torch.zeros(count, dtype=torch.long, device=device)
    current = states[:, 0]
    for step in range(horizon):
        following = apply_delta(current, model_state(model, current, control[:, step]))
        valid = torch.isfinite(following).all(-1) & (following.abs().amax(-1) <= 1e4)
        active = active & valid
        completed += active
        predictions[:, step + 1] = torch.where(active[:, None], following, current)
        # Retired members use a finite state and are masked from all metrics.
        current = torch.where(active[:, None], following, states[:, 0])
    return one_step, predictions.cpu().numpy(), completed.cpu().numpy()


def diagnostics(sim, predicted, truth, trajectory, realized):
    penetration, contacts = [], []
    for state in predicted[1:realized + 1]:
        sim.reset(state)
        values = sim.diagnostics()
        penetration.append(float(values['penetration']))
        contacts.append(values['contact_count'] > 0)
    true_contact = trajectory['contact_count'][1:realized + 1] > 0
    predicted_contact = np.asarray(contacts, dtype=bool)
    true_onsets = np.flatnonzero(true_contact & ~np.r_[False, true_contact[:-1]]) if realized else np.array([], dtype=int)
    predicted_onsets = np.flatnonzero(predicted_contact & ~np.r_[False, predicted_contact[:-1]]) if realized else np.array([], dtype=int)
    onset = float(np.mean([np.min(np.abs(predicted_onsets - i)) * sim.dt for i in true_onsets])) if len(true_onsets) and len(predicted_onsets) else None
    return {
        'penetration_max': max(penetration, default=0.),
        'penetration_mean': float(np.mean(penetration)) if penetration else 0.,
        'contact_disagreement': float(np.mean(predicted_contact != true_contact)) if realized else None,
        'contact_onset_error_seconds': onset, 'contact_onset_resolution_seconds': sim.dt,
        'ground_truth_substep_penetration_max': float(np.max(trajectory.get('substep_penetration', trajectory['penetration']))),
        'missed_contact_onsets': int(sum(not len(predicted_onsets) or np.min(np.abs(predicted_onsets - i)) > 2 for i in true_onsets)),
        'false_contact_onsets': int(sum(not len(true_onsets) or np.min(np.abs(true_onsets - i)) > 2 for i in predicted_onsets)),
        'onset_match_tolerance_steps': 2,
    }


def aggregate(records):
    result = {}
    for group in ('one_step', 'rollout', 'impact', 'persistent_contact'):
        valid = [row[group] for row in records if row.get(group) is not None]
        result['mean_' + group] = {key: float(np.mean([row[key] for row in valid])) for key in valid[0]} if valid else {}
    fields = ('penetration_mean', 'penetration_max', 'contact_disagreement', 'contact_onset_error_seconds', 'missed_contact_onsets', 'false_contact_onsets')
    result['mean_diagnostics'] = {key: float(np.mean(values)) if (values := [row[key] for row in records if row.get(key) is not None]) else None for key in fields}
    result['metric_trajectory_counts'] = {group: sum(row.get(group) is not None for row in records) for group in ('rollout', 'impact', 'persistent_contact')}
    result['divergence_rate'] = float(np.mean([row['diverged'] for row in records]))
    result['rollout_steps'] = {'min': min(row['rollout_steps'] for row in records), 'max': max(row['rollout_steps'] for row in records)}
    return result


def evaluate_model(name, checkpoint, dataset_path, out, batch_size, transition_batch_size, device, verify):
    root, manifest = load_manifest(dataset_path)
    entries = [entry for entry in manifest['trajectories'] if entry['split'] == 'ood']
    groups = defaultdict(list)
    for entry in entries:
        groups[scene_key(entry)].append(entry)
    model, payload = load_checkpoint(str(checkpoint), device)
    model.float().eval()
    if int(model.config.get('history', 1)) != 1:
        raise ValueError('This evaluator requires history=1')
    assert payload['epoch'] == 5 and payload['seed'] == 0
    if model.kind != name:
        raise ValueError(f'Expected {name} checkpoint, got {model.kind}')
    torch.set_num_threads(1)
    cache, records, equivalence = SceneCache(4), [], []
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = {'model_name': name, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                'dataset': dataset_path, 'dataset_fingerprint': manifest['dataset_fingerprint'],
                'dataset_manifest_sha256': digest(root / 'manifest.json'),
                'split': 'ood', 'device': device, 'seed': payload['seed'], 'epoch': payload['epoch'],
                'checkpoint_meta': {key: payload.get(key) for key in ('epoch', 'seed', 'global_step', 'val_loss', 'model_config')},
                'trajectory_batch_size': batch_size, 'one_step_batch_size': transition_batch_size,
                'available_trajectories': len(entries), 'save_rollouts': False,
                'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                'evaluation_engine': 'same-scene batched FP32; official state_metrics and per-trajectory arithmetic means; post-inference MuJoCo diagnostics',
                'not_measured': ['single-rollout latency', 'initial local pair coverage'],
                'code_sha256': {str(path): digest(path) for path in (Path(__file__), Path('src/cube/models/geometry.py'), Path('src/cube/models/contact.py'), Path('src/cube/models/dynamics.py'), Path('src/cube/models/state.py'), Path('src/cube/models/serialization.py'), Path('src/cube/experiments/common.py'))},
                'complete': False}
    write_json(out / 'evaluation.json', metadata)
    with torch.inference_mode(), (out / 'trajectories.jsonl').open('w', encoding='utf-8') as stream:
        for group in groups.values():
            sim = cache.activate(model, group[0])
            if verify:
                pilot = group[:batch_size]
                arrays = [load_trajectory(root, e) for e in pilot]
                one, batched, completed = infer(model, np.stack([a['states'] for a in arrays]),
                                               np.stack([a['actions'] for a in arrays]), device, transition_batch_size)
                reference = predict_trajectory(str(checkpoint), pilot[0]['scene_spec'], arrays[0]['states'], arrays[0]['actions'],
                                               horizon=100, device=device, progress_config=False, loaded_checkpoint=(model, payload))
                n = reference['completed_steps']
                expected = reference['predicted']
                n_batched = int(completed[0])
                common = min(n, n_batched)
                observed = batched[0, :common + 1]
                expected = expected[:common + 1]
                difference = float(np.max(np.abs(observed - expected)))
                # Batch-shaped GEMMs/reductions can amplify FP32 roundoff in
                # a long unstable rollout. Check fixed-GT one-step outputs
                # strictly, and expose the long-rollout differences as data.
                sequential_one = []
                for offset in range(0, 100, 8):
                    x = torch.as_tensor(arrays[0]['states'][offset:offset + 8].copy(), dtype=torch.float32, device=device)
                    u = torch.as_tensor(arrays[0]['actions'][offset:offset + 8].copy(), dtype=torch.float32, device=device)
                    if len(x) > len(u):
                        x = x[:len(u)]
                    sequential_one.append(apply_delta(x, model_state(model, x, u)).cpu().numpy())
                sequential_one = np.concatenate(sequential_one)
                np.testing.assert_allclose(one[0], sequential_one, rtol=2e-5, atol=2e-5,
                                           err_msg=f'Batch vs official one-step mismatch: {name}/{pilot[0]["scenario"]}')
                observed_metrics = state_metrics(observed[1:], arrays[0]['states'][1:common + 1]) if common else {}
                expected_metrics = state_metrics(expected[1:], arrays[0]['states'][1:common + 1]) if common else {}
                equivalence.append({'trajectory_id': pilot[0]['id'], 'scenario': pilot[0]['scenario'],
                                    'production_trajectory_batch_size': len(pilot),
                                    'reference_completed_steps': n, 'batched_completed_steps': n_batched,
                                    'maximum_state_absolute_difference': difference,
                                    'strict_100_step_state_allclose': bool(n == n_batched and np.allclose(observed, expected, rtol=2e-3, atol=2e-3)),
                                    'one_step_maximum_absolute_difference': float(np.max(np.abs(one[0] - sequential_one))),
                                    'one_step_comparison_passed': True,
                                    'rollout_metric_absolute_differences': {key: abs(observed_metrics[key] - expected_metrics[key]) for key in observed_metrics},
                                    'reference': 'existing FP32 sequential rollout; official per-trajectory one-step batch=8',
                                    'note': 'Long-rollout FP32 numerical sensitivity is reported, not treated as bitwise equivalence.',
                                    'passed': True})
                sim = cache.activate(model, group[0])
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
                arrays = [load_trajectory(root, e) for e in batch]
                truth, controls = np.stack([a['states'] for a in arrays]), np.stack([a['actions'] for a in arrays])
                inference_started = time.perf_counter()
                one_step, predictions, completed = infer(model, truth, controls, device, transition_batch_size)
                inference_seconds = time.perf_counter() - inference_started
                for index, (entry, trajectory) in enumerate(zip(batch, arrays)):
                    n = int(completed[index])
                    contact = trajectory['contact_count'][1:] > 0
                    impact = trajectory.get('impact_event', contact & (trajectory['contact_count'][:-1] == 0)).astype(bool)
                    persistent = trajectory.get('persistent_contact', contact & (trajectory['contact_count'][:-1] > 0)).astype(bool)
                    following = truth[index, 1:]
                    row = {'trajectory_id': entry['id'], 'scenario': entry['scenario'], 'split': 'ood',
                           'one_step': state_metrics(one_step[index], following),
                           'impact': state_metrics(one_step[index, impact], following[impact]) if impact.any() else None,
                           'persistent_contact': state_metrics(one_step[index, persistent], following[persistent]) if persistent.any() else None,
                           'impact_frames': int(impact.sum()), 'persistent_frames': int(persistent.sum()),
                           'event_label_source': 'substep contacts and approach velocity',
                           'rollout': state_metrics(predictions[index, 1:n + 1], following[:n]) if n else None,
                           'rollout_steps': n, 'requested_rollout_horizon': 100, 'diverged': n < 100,
                           **diagnostics(sim, predictions[index], truth[index], trajectory, n)}
                    records.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                progress = {'model': name, 'scenario': batch[0]['scenario'], 'evaluated': len(records),
                            'total': len(entries), 'inference_batch_seconds': round(inference_seconds, 3),
                            'elapsed_seconds': round(time.perf_counter() - started, 1)}
                write_json(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    result = {**metadata, **aggregate(records), 'complete': True, 'evaluated_trajectories': len(records),
              'trajectory_ids': [row['trajectory_id'] for row in records], 'trajectories': records,
              'trajectory_metrics_file': str(out / 'trajectories.jsonl'),
              'scenarios': {s: {'evaluated_trajectories': len(selected), **aggregate(selected)}
                            for s in sorted({r['scenario'] for r in records})
                            if (selected := [r for r in records if r['scenario'] == s])},
              'batch_equivalence_checks': equivalence, 'elapsed_seconds': time.perf_counter() - started}
    write_json(out / 'evaluation.json', result)
    print(json.dumps({'model': name, 'complete': True, 'evaluated': len(records),
                      'rollout': result['mean_rollout'], 'divergence_rate': result['divergence_rate']}), flush=True)
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/cube-toss-ood-v2')
    parser.add_argument('--output', default='outputs/ood-v2-evaluation')
    parser.add_argument('--models', nargs='+', choices=('state_mlp', 'deepsets', 'hierarchical', 'contact_mlp'), default=['state_mlp', 'deepsets', 'hierarchical', 'contact_mlp'])
    parser.add_argument('--contact-checkpoint', default='outputs/contact-mlp-5ep/best.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--transition-batch-size', type=int, default=32)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.transition_batch_size < 1:
        parser.error('Batch sizes must be positive')
    output = Path(args.output)
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        checkpoint = inputs / f'{name}-best.pt'
        original = (Path(args.contact_checkpoint) if name == 'contact_mlp' else
                    Path(f'outputs/cube-toss-suite-final/{name}_budget10000_seed0/best.pt'))
        report_path = output / name / 'evaluation.json'
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding='utf-8'))
            if previous.get('complete'):
                _, manifest = load_manifest(args.dataset)
                if previous['dataset_fingerprint'] != manifest['dataset_fingerprint'] or previous['checkpoint_sha256'] != digest(original):
                    raise ValueError('Completed evaluation identity changed; use a new output directory')
                print(json.dumps({'model': name, 'reused_complete': True}), flush=True)
                continue
        shutil.copy2(original, checkpoint)
        evaluate_model(name, checkpoint, args.dataset, output / name, args.batch_size,
                       args.transition_batch_size, args.device, args.verify)


if __name__ == '__main__':
    main()
