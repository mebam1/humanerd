"""Command-line entry points; paths are relative to the mounted repository root."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cube → two-link → Ant contact dynamics research")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print installed package versions and runtime availability")
    sub.add_parser("docs", help="Build the local HTML documentation")
    p = sub.add_parser("quick", help="Run the small experiments and generate the measured HTML report")
    p.add_argument("--output", default="outputs/quick")
    p.add_argument("--with-warp", action="store_true")
    p = sub.add_parser("report", help="Rebuild figures and report from saved quick-experiment results")
    p.add_argument("--output", default="outputs/quick")
    p = sub.add_parser("render", help="Render stored states with headless MuJoCo")
    p.add_argument("--trajectory", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--key", default="states")
    p.add_argument("--stride", type=int, default=1)
    p = sub.add_parser("list-trajectories", help="List dataset trajectory IDs without loading all state arrays")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["train", "val", "test", "ood"])
    p.add_argument("--scenario")
    p.add_argument("--trajectory-id", type=int)
    p.add_argument("--limit", type=int, default=10)
    for command, description in (("visualize-data", "Render selected GT trajectories from NPZ or sharded datasets"),
                                 ("visualize-rollout", "Render a learned autonomous rollout beside synchronized GT")):
        p = sub.add_parser(command, help=description)
        p.add_argument("--dataset", required=True)
        p.add_argument("--split", choices=["train", "val", "test", "ood"])
        p.add_argument("--scenario")
        p.add_argument("--trajectory-id", type=int)
        p.add_argument("--limit", type=int, default=4)
        p.add_argument("--output", required=True, help="Gallery directory; put under docs/ to view at localhost")
        p.add_argument("--width", type=int, default=640, help="Width of each video panel")
        p.add_argument("--height", type=int, default=480)
        p.add_argument("--stride", type=int, default=1)
        p.add_argument("--horizon", type=int, default=100 if command == "visualize-rollout" else None)
        if command == "visualize-rollout":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--device", default="cpu", help="Model inference device; rendering uses MUJOCO_GL independently")
    p = sub.add_parser("generate", help="Generate trajectory-split simulator data")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", help="Continue only a dataset with matching generation settings")
    p = sub.add_parser("data-plan", help="Inspect dataset counts and disk estimate without generating")
    p.add_argument("--config", required=True)
    p = sub.add_parser("train", help="Train a dynamics model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config")
    p.add_argument("--model", dest="model_name", default="hierarchical")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume", help="Resume optimizer/RNG/scheduler from last.pt")
    p.add_argument("--device", help="Training device override, e.g. cuda:0")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-entity")
    p.add_argument("--transfer-checkpoint")
    p.add_argument("--transfer-mode", default="scratch", choices=["scratch", "full", "point-frozen", "aggregation-frozen"])
    p = sub.add_parser("evaluate", help="One-step and autonomous rollout metrics")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--rollout-horizon", type=int, default=32)
    p.add_argument("--device", help="Inference device (defaults to checkpoint configuration)")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-trajectories", type=int)
    p.add_argument("--no-save-rollouts", dest="save_rollouts", action="store_false", default=None)
    p = sub.add_parser("validate", help="Check simulator state conventions and replay")
    p.add_argument("--output", default="outputs/validation")
    p = sub.add_parser("coupling", help="Matched empty/A/B/AB contact interventions")
    p.add_argument("--output", default="outputs/coupling")
    p.add_argument("--checkpoint")
    p = sub.add_parser("control", help="Evaluate a controller in independent simulator physics")
    p.add_argument("--output", required=True)
    p.add_argument("--kind", default="ant", choices=["cube", "ant", "two_link"])
    p.add_argument("--checkpoint")
    p.add_argument("--controller", default="baseline", choices=["baseline", "mpc", "zero"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--video", action="store_true")
    p.add_argument("--goal", nargs=3, type=float)
    p = sub.add_parser("suite", help="Run a configured multi-model experiment suite")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", help="Skip completed runs and resume incomplete ones")
    p.add_argument("--models", dest="model_filter", nargs="+", help="Run only these model labels from the configured suite")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    p = sub.add_parser("transfer", help="Compare adaptation modes at target data budgets")
    p.add_argument("--config")
    p.add_argument("--output", required=True)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--target-dataset", required=True)
    for name in ("generate", "data-plan", "validate", "coupling", "control", "suite"):
        command = sub.choices[name]
        command.add_argument("--backend", choices=["mujoco", "warp"], default="mujoco")
        command.add_argument("--device", default="cpu", help="Simulator device: cpu or cuda:0")
    args = vars(parser.parse_args(argv))
    name = args.pop("command")
    if name == "doctor":
        import torch
        import warp as wp
        wp.init()
        result = {"packages": {p: importlib.metadata.version(p) for p in
                  ["numpy", "torch", "mujoco", "mujoco-warp", "warp-lang", "wandb", "tqdm"]},
                  "torch_cuda": torch.cuda.is_available(),
                  "torch_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "wandb_key_configured": bool(__import__("os").environ.get("WANDB_API_KEY")),
                  "warp_devices": [str(d) for d in wp.get_devices()],
                  "workspace": str(Path.cwd())}
    elif name == "docs":
        from cube.documentation import build_docs
        result = build_docs()
    elif name == "render":
        from cube.rendering import render_trajectory
        result = render_trajectory(**args)
    elif name == "list-trajectories":
        from cube.visualization_data import list_trajectories
        result = list_trajectories(**args)
    elif name in {"visualize-data", "visualize-rollout"}:
        from cube.visualization import visualize_data, visualize_rollout
        result = (visualize_data if name == "visualize-data" else visualize_rollout)(**args)
    elif name == "quick":
        from cube.quick import run_quick
        result = run_quick(**args)
    elif name == "report":
        from cube.reporting import build_report
        result = build_report(**args)
    elif name == "data-plan":
        from cube.experiments.data import dataset_plan
        result = dataset_plan(**args)
    else:
        from cube import experiments
        if name in {"train", "suite"}:
            from cube.experiments.common import read_config
            cfg = read_config(args["config"])
            destination = cfg if name == "train" else cfg.setdefault("training", {})
            for field in ("mode", "project", "entity"):
                value = args.pop(f"wandb_{field}", None)
                if value is not None:
                    destination.setdefault("wandb", {})[field] = value
            if name == "train":
                device = args.pop("device", None)
                if device is not None:
                    destination["device"] = device
            args["config"] = cfg
        fn = getattr(experiments, {"generate": "generate_dataset", "suite": "run_suite"}.get(name, name))
        result = fn(**args)
    # Detailed arrays remain in result files; terminal summaries stay readable.
    if isinstance(result, dict) and name not in {"doctor", "list-trajectories"}:
        result = {key: (f"{len(value)} records (see output JSON)" if isinstance(value, list)
                       else value) for key, value in result.items()
                  if key not in {"normalization", "transfer", "step_records"}}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
