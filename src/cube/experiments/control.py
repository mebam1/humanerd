"""Feedback gait and learned-model shooting MPC evaluated in the real simulator."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .common import history_window, make_sim, model_state, sim_state, upright, write_json
from .learning import load_checkpoint


def ant_gait_action(state: np.ndarray, time_seconds: float, frequency: float = 1.4,
                    hip_amplitude: float = 0.35, knee_lift: float = 0.4) -> np.ndarray:
    """Diagonal trot: swing retracts knees; stance sweeps feet toward the rear.

    This is a reproducible hand-designed PD gait, not a learned policy. The
    denominator is the MJCF motor gear (20 Nm per normalized control unit).
    """
    state = np.asarray(state)
    if state.shape[-1] != 29:
        raise ValueError("The Ant gait expects a floating base and eight hinge joints")
    phase = 2 * np.pi * frequency * time_seconds + np.array([0.0, np.pi, np.pi, 0.0])
    side = np.array([1, -1, 1, -1])
    targets = np.zeros(8)
    targets[::2] = -side * hip_amplitude * np.sin(phase)
    targets[1::2] = -knee_lift * np.maximum(np.sin(phase), 0.0)
    error = targets - state[7:15]
    torques = 12.0 * error - 0.8 * state[21:29]
    return np.clip(torques / 20.0, -1.0, 1.0)


def _nominal_action(state, kind, now, goal):
    if kind == "ant":
        return ant_gait_action(state, now)
    if kind == "two_link":
        return np.clip(-0.2 * state[7:8] - 0.03 * state[-1:], -1, 1)
    nq = (len(state)+1)//2
    action = np.zeros(6)
    action[:2] = np.clip(40 * (goal[:2]-state[:2]) - 8 * state[nq:nq+2], -20, 20)
    action[3:] = np.clip(-0.2 * state[nq+3:nq+6], -2, 2)
    return action


def _cost(states, actions, kind, goal):
    nq = (states.shape[-1]+1)//2
    if kind == "ant":
        # Forward locomotion, an upright trunk, safe height and low control effort.
        tilt = 1 - 2 * (states[:, 4].square() + states[:, 5].square())
        return (-states[:, nq] + 2*(states[:, 2]-0.34).square() + 2*(1-tilt).square()
                + 0.1*states[:, 1].square() + 0.02*actions.square().sum(-1)
                + 20*torch.relu(0.15-states[:, 2]))
    target = torch.as_tensor(goal, dtype=states.dtype, device=states.device)
    return (states[:, :3]-target).square().sum(-1) + 0.1*states[:, nq:nq+3].square().sum(-1) + 0.01*actions.square().sum(-1)


def shooting_mpc(model, state: np.ndarray, history: np.ndarray, kind: str, time_seconds: float, dt: float,
                 rng: np.random.Generator, goal: np.ndarray, horizon: int = 6, candidates: int = 16,
                 iterations: int = 2, noise_scale: float = 0.15, action_history: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    from cube.models import apply_delta
    if min(horizon, candidates, iterations) < 1:
        raise ValueError("MPC horizon, candidates and iterations must be positive")
    action_dim = 8 if kind == "ant" else 1 if kind == "two_link" else 6
    means = np.asarray([_nominal_action(state, kind, time_seconds+i*dt, goal) for i in range(horizon)])
    limit = np.ones(action_dim) if kind in {"ant", "two_link"} else np.array([20., 20., 20., 2., 2., 2.])
    std = np.broadcast_to(noise_scale * limit, (horizon, action_dim)).copy()
    started = time.perf_counter()
    best_cost = float("inf")
    best_actions = means.copy()
    with torch.no_grad():
        for _ in range(iterations):
            plans = np.clip(rng.normal(means, std, size=(candidates, horizon, action_dim)), -limit, limit)
            plans[0] = means  # Always retain the current nominal proposal.
            current = torch.as_tensor(np.repeat(state[None], candidates, axis=0), dtype=torch.float32)
            histories = torch.as_tensor(np.repeat(history[None], candidates, axis=0), dtype=torch.float32)
            initial_actions = np.zeros((len(history), action_dim)) if action_history is None else action_history
            histories_actions = torch.as_tensor(np.repeat(initial_actions[None], candidates, axis=0), dtype=torch.float32)
            costs = torch.zeros(candidates)
            for step in range(horizon):
                action = torch.as_tensor(plans[:, step], dtype=torch.float32)
                histories_actions = torch.cat((histories_actions[:, :-1], action[:, None]), dim=1)
                current = apply_delta(current, model_state(model, current, action, {"states": histories, "actions": histories_actions}))
                costs = costs + _cost(current, action, kind, goal)
                histories = torch.cat((histories[:, 1:], current[:, None]), dim=1)
                histories_actions = torch.cat((histories_actions[:, 1:], torch.zeros_like(action[:, None])), dim=1)
            values = torch.nan_to_num(costs, nan=1e12, posinf=1e12, neginf=1e12).numpy()
            elite_indices = np.argsort(values)[:max(1, candidates//4)]
            means = plans[elite_indices].mean(0)
            std = plans[elite_indices].std(0).clip(min=0.02)
            if float(values[elite_indices[0]]) < best_cost:
                best_cost = float(values[elite_indices[0]])
                best_actions = plans[elite_indices[0]]
    return best_actions[0], {"predicted_cost": best_cost, "planning_ms": (time.perf_counter()-started)*1000,
                             "model_transitions": candidates*horizon*iterations}


def control(output: str, kind: str = "cube", checkpoint: str | None = None, controller: str = "baseline",
            steps: int = 200, backend: str = "mujoco", device: str = "cpu", seed: int = 0,
            horizon: int = 6, candidates: int = 16, iterations: int = 2, goal: list | None = None,
            video: bool = False) -> dict:
    if controller not in {"baseline", "mpc", "zero"}:
        raise ValueError("controller must be baseline, mpc, or zero")
    if controller == "mpc" and checkpoint is None:
        raise ValueError("MPC requires a trained dynamics checkpoint")
    if steps < 1:
        raise ValueError("steps must be positive")
    spec = {"kind": kind, "scenario": "walk" if kind == "ant" else "rest", "seed": seed}
    sim = make_sim(spec, backend, device)
    initial = sim_state(sim)
    target = np.asarray(goal if goal is not None else [0.3, 0, initial[2]], dtype=float)
    model, payload = load_checkpoint(checkpoint) if checkpoint else (None, None)
    if model is not None:
        torch.set_num_threads(int(payload["training_config"].get("torch_threads", 1)))
        model.set_scene(sim)
    history_length = int(model.config.get("history", 1)) if model is not None else 1
    rng = np.random.default_rng(seed)
    states, actions, records, frames = [initial], [], [], []
    renderer = None
    if video:
        import mujoco
        renderer = mujoco.Renderer(sim.model, height=360, width=480)
    for i in range(steps):
        if controller == "mpc":
            action, diagnostic = shooting_mpc(model, states[-1], history_window(np.asarray(states), i, history_length),
                                              kind, i*sim.dt, sim.dt, rng, target, horizon, candidates, iterations,
                                              action_history=history_window(np.asarray(actions + [np.zeros(sim.action_dim)]), i, history_length))
        else:
            started = time.perf_counter()
            action = np.zeros(sim.action_dim) if controller == "zero" else _nominal_action(states[-1], kind, i*sim.dt, target)
            diagnostic = {"planning_ms": (time.perf_counter()-started)*1000, "model_transitions": 0}
        next_state = np.asarray(sim.step(action))
        if not np.isfinite(next_state).all():
            raise RuntimeError("Ground-truth simulator produced a non-finite control state")
        states.append(next_state.copy())
        actions.append(action)
        records.append({**diagnostic, **sim.diagnostics()})
        if renderer is not None:
            renderer.update_scene(sim.data)
            frames.append(renderer.render())
    states, actions = np.asarray(states), np.asarray(actions)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "rollout.npz", states=states, actions=actions)
    write_json(output_dir / "scene.json", sim.spec.to_dict())
    if renderer is not None:
        renderer.close()
        import imageio.v3 as iio
        iio.imwrite(output_dir / "rollout.gif", np.asarray(frames), duration=int(1000*sim.dt), loop=0)
    displacement = float(states[-1, 0]-states[0, 0])
    fallen = (states[:, 2] < 0.14) | (upright(states) < 0.3) if kind == "ant" else np.zeros(len(states), dtype=bool)
    success = bool(displacement > 0.15 and not fallen.any()) if kind == "ant" else bool(np.linalg.norm(states[-1,:3]-target) < 0.1)
    with torch.no_grad():
        costs = _cost(torch.tensor(states[1:], dtype=torch.float32), torch.tensor(actions, dtype=torch.float32), kind, target).numpy()
    report = {"kind": kind, "controller": controller, "checkpoint": checkpoint, "backend": backend, "seed": seed,
              "steps": steps, "duration_seconds": steps*sim.dt, "goal": target, "return": float(-costs.sum()),
              "success": success, "forward_displacement": displacement, "average_forward_speed": displacement/(steps*sim.dt),
              "fall_fraction": float(fallen.mean()), "final_goal_distance": float(np.linalg.norm(states[-1,:3]-target)),
              "action_squared_mean": float(np.mean(actions**2)), "penetration_max": max(float(r["penetration"]) for r in records),
              "planning_mean_ms": float(np.mean([r["planning_ms"] for r in records])),
              "mpc_budget": {"horizon": horizon, "candidates": candidates, "iterations": iterations} if controller == "mpc" else None,
              "ground_truth_evaluation": True, "step_records": records,
              "notice": "baseline is a hand-designed gait; MPC uses a fitted dynamics model. Success is measured, never assumed."}
    return write_json(output_dir / "control.json", report)
