"""Bounded GPU capacity probe; measures update speed, not predictive accuracy.

Run through Compose, for example:
docker compose run --rm cuda python scripts/benchmark_gpu.py --config configs/train_gpu.json
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch

from cube.experiments.common import make_sim, read_config, sim_state, write_json
from cube.experiments.learning import _prediction_loss, _seed
from cube.experiments.training_runtime import _amp_context, resolve_device
from cube.models import build_model, state_delta


VARIANTS = {"hierarchical": {"name": "hierarchical"}, "state_mlp": {"name": "state_mlp"},
            "deepsets": {"name": "deepsets"}, "flat": {"name": "flat"},
            "independent": {"name": "independent"},
            "nerd_history1": {"name": "nerd_like", "history": 1},
            "nerd_history10": {"name": "nerd_like", "history": 10}}


def measure(cfg, variant, batch_size, warmup, steps, scenario):
    device = resolve_device(cfg.get("device", "cuda:0"))
    precision = cfg.get("mixed_precision", "bf16")
    model_config = {**cfg.get("model", {}), **VARIANTS[variant]}
    sim = make_sim({"kind": "cube", "scenario": scenario, "seed": 12})
    initial = sim_state(sim)
    action = np.zeros(6, dtype=np.float32)
    following = sim.step(action)
    model = build_model(sim, model_config).to(device).train()
    state = torch.tensor(initial, device=device, dtype=torch.float32)[None].repeat(batch_size, 1)
    actions = torch.tensor(action, device=device)[None].repeat(batch_size, 1)
    truth = torch.tensor(following, device=device, dtype=torch.float32)[None].repeat(batch_size, 1)
    target = state_delta(state, truth)
    history_length = model.config["history"]
    history = {"states": state[:, None].repeat(1, history_length, 1),
               "actions": actions[:, None].repeat(1, history_length, 1)}
    scale = torch.ones_like(target[0]).clamp_min(.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", .001)))
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    def update():
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, precision):
            prediction = model(state, actions, history)
        loss, _ = _prediction_loss(prediction.float(), target, state, truth, scale)
        if not torch.isfinite(loss):
            raise RuntimeError("Capacity probe produced a non-finite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    for _ in range(warmup):
        update()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(steps):
        loss = update()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    return {"model": variant, "batch_size": batch_size, "status": "ok", "scenario": scenario,
            "point_count": len(model.features.local_points), "parameters": sum(p.numel() for p in model.parameters()),
            "measured_steps": steps, "warmup_steps": warmup, "milliseconds_per_microbatch_update": 1000*elapsed/steps,
            "samples_per_second": batch_size*steps/elapsed,
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device)/2**20,
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device)/2**20,
            "last_probe_loss": loss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--output", default="outputs/gpu_capacity.json")
    parser.add_argument("--models", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--scenario", choices=["impact", "wall_only", "floor_wall"], default="wall_only")
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("warmup must be nonnegative and steps positive")
    cfg = read_config(args.config)
    batch_size = args.batch_size or int(cfg.get("batch_size", 8))
    if batch_size < 1:
        parser.error("batch size must be positive")
    _seed(12)
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    device = resolve_device(cfg.get("device", "cuda:0"))
    rows = []
    report = {"gpu": torch.cuda.get_device_name(device),
              "gpu_total_memory_mb": torch.cuda.get_device_properties(device).total_memory/2**20,
              "torch_version": torch.__version__, "precision": cfg.get("mixed_precision", "bf16"),
              "records": rows,
              "notice": "Short capacity probe on repeated simulator states; throughput and memory only, not accuracy or convergence. Includes one optimizer update per microbatch; full training accumulates gradients."}
    for variant in args.models:
        candidates = [batch_size] + ([2] if variant == "independent" and batch_size > 2 else [])
        for size in candidates:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                row = measure(cfg, variant, size, args.warmup, args.steps, args.scenario)
                rows.append(row)
                print(f"{variant}: batch={size}, {row['samples_per_second']:.2f} samples/s, "
                      f"peak={row['gpu_peak_allocated_mb']:.0f} MiB", flush=True)
                break
            except torch.OutOfMemoryError:
                rows.append({"model": variant, "batch_size": size, "status": "out_of_memory", "scenario": args.scenario})
                print(f"{variant}: batch={size} out of memory", flush=True)
            finally:
                write_json(Path(args.output), report)
    return report


if __name__ == "__main__":
    main()
