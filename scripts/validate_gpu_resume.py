"""Exercise real CUDA AMP, offline W&B SDK and mid-epoch optimizer restoration."""
from pathlib import Path

import torch

from cube.experiments import train
from cube.experiments.common import read_config, write_json


def main():
    root = Path("outputs/gpu-resume-check")
    config = read_config("configs/train_gpu.json")
    config.update(epochs=2, trajectory_budget=4, validation_max_trajectories=2,
                  batch_size=16, gradient_accumulation_steps=2, max_steps=2,
                  progress={"enabled": False}, checkpoint_every_steps=1)
    config["wandb"] = {"mode": "offline", "group": "gpu-resume-check"}
    baseline = train("data/cube-nerd", str(root / "uninterrupted"), config, "state_mlp", seed=42)
    first = train("data/cube-nerd", str(root / "resumed"), {**config, "max_steps": 1}, "state_mlp", seed=42)
    resumed = train("data/cube-nerd", str(root / "resumed"), config, "state_mlp", seed=42,
                    resume=first["last_checkpoint"])
    left = torch.load(baseline["last_checkpoint"], map_location="cpu", weights_only=False)
    right = torch.load(resumed["last_checkpoint"], map_location="cpu", weights_only=False)
    maximum = max(float((left["state_dict"][key]-right["state_dict"][key]).abs().max())
                  for key in left["state_dict"])
    assert maximum == 0., maximum
    assert left["scheduler_state_dict"] == right["scheduler_state_dict"]
    assert first["wandb"]["id"] == resumed["wandb"]["id"]
    assert left["global_step"] == right["global_step"] == 2
    result = {"gpu": torch.cuda.get_device_name(), "precision": config["mixed_precision"],
              "optimizer_steps": 2, "resumed_max_weight_difference": maximum,
              "scheduler_state_equal": True, "offline_wandb_id_preserved": True,
              "wandb_mode": "offline", "online_verified": False,
              "scope": "Runtime restoration check; not an accuracy experiment"}
    write_json(root / "verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
