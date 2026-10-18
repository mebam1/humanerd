"""Explicit experiment telemetry; online requests never silently become offline runs."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from tqdm.auto import tqdm


def progress(iterable=None, *, config=None, **kwargs):
    """TTY-aware tqdm, overridable by config or TQDM_DISABLE/TQDM_FORCE."""
    config = config or {}
    if isinstance(config, bool):
        config = {"enabled": config}
    disabled = os.environ.get("TQDM_DISABLE", "").lower() in {"1", "true", "yes"}
    forced = os.environ.get("TQDM_FORCE", "").lower() in {"1", "true", "yes"}
    enabled = config.get("enabled", sys.stderr.isatty() or forced)
    return tqdm(iterable, disable=disabled or not enabled,
                mininterval=float(config.get("mininterval", 1.0)), dynamic_ncols=True, **kwargs)


class RunTracker:
    """A small lazy SDK boundary, making disabled training credential/network free."""

    def __init__(self, config: dict, run_config: dict, output: Path, run_id: str | None = None):
        self.run = None
        self.mode = config.get("mode", "disabled")
        self.id = None
        self.url = None
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.mode == "disabled":
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B logging requires the wandb dependency; rebuild the Docker image") from error
        # Config is an explicit, caller-built allowlist. No os.environ, API key,
        # arbitrary training configuration, source tree, dataset or artifact upload.
        kwargs = {key: config[key] for key in ("project", "entity", "group", "name", "tags") if config.get(key)}
        kwargs.setdefault("project", os.environ.get("WANDB_PROJECT") or "cube-contact")
        try:
            self.run = wandb.init(**kwargs, mode=self.mode, dir=str(output), config=run_config,
                                  id=run_id, resume="must" if run_id and self.mode == "online" else None,
                                  settings=wandb.Settings(disable_git=True, save_code=False))
            if self.run is None:
                raise RuntimeError("wandb.init returned no run")
            actual_mode = getattr(getattr(self.run, "settings", None), "mode", self.mode)
            if self.mode == "online" and actual_mode != "online":
                raise RuntimeError(f"W&B online mode was requested but SDK started in {actual_mode!r} mode")
            self.id = self.run.id
            self.url = self.run.url if self.mode == "online" else None
            self.run.define_metric("optimizer_step")
            self.run.define_metric("train/*", step_metric="optimizer_step")
            self.run.define_metric("epoch")
            self.run.define_metric("epoch/*", step_metric="epoch")
        except Exception as error:
            if self.run is not None:
                self.run.finish(exit_code=1)
            raise RuntimeError(f"W&B {self.mode} initialization failed. Check WANDB_API_KEY, project/entity, and network. "
                               "The requested logging mode was not silently changed.") from error

    def log(self, values: dict):
        if self.run is not None:
            self.run.log(values)

    def finish(self, summary: dict | None = None, *, failed: bool = False):
        if self.run is not None:
            if summary:
                self.run.summary.update(summary)
            self.run.finish(exit_code=1 if failed else 0)

