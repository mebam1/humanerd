"""Reproduce the small, explicitly non-publication-scale experiments."""
from __future__ import annotations

import importlib.metadata
import hashlib
from pathlib import Path
import platform
import time
import runpy
from datetime import datetime, timedelta, timezone

from cube.experiments import control, coupling, evaluate, generate_dataset, run_suite, train, transfer, validate
from cube.experiments.common import write_json


def run_quick(output="outputs/quick", with_warp=False):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    def note(message):
        print(message, flush=True)
    note("[1/8] Physics conventions and deterministic replay")
    validate(str(root / "validation"))
    if with_warp:
        validate(str(root / "warp_validation"), backend="warp", device="cuda:0")
    note("[2/8] 12 matched Cube runs: 3 models × 2 budgets × 2 seeds")
    run_suite("configs/suite_smoke.json", str(root / "suite"))
    source = str(root / "suite/hierarchical_budget8_seed0/best.pt")
    note("[3/8] Same-face and cross-face contact interventions")
    coupling(str(root / "coupling"))
    coupling(str(root / "coupling_model"), checkpoint=source)
    note("[4/8] Cube to two-link: 4 transfer modes × 2 target budgets")
    generate_dataset("configs/two_link_smoke.json", str(root / "two_link_dataset"))
    transfer(source, str(root / "two_link_dataset"), str(root / "transfer"), "configs/transfer_smoke.json")
    note("[5/8] Ant dynamics: 3-epoch pipeline validation")
    generate_dataset("configs/ant_smoke.json", str(root / "ant_dataset"))
    ant = train(str(root / "ant_dataset"), str(root / "ant_model"), "configs/train_smoke.json")
    evaluate(ant["checkpoint"], str(root / "ant_dataset"), str(root / "ant_evaluation"), rollout_horizon=16)
    note("[6/8] Ant control: 6s gait/zero; matched 1s gait/MPC")
    control(str(root / "ant_baseline"), kind="ant", steps=300)
    control(str(root / "ant_zero"), kind="ant", controller="zero", steps=300)
    control(str(root / "ant_baseline_short"), kind="ant", steps=50)
    control(str(root / "ant_mpc"), kind="ant", controller="mpc", checkpoint=ant["checkpoint"], steps=50,
            horizon=3, candidates=8, iterations=1)
    note("[7/8] Cube force-control: matched 1s baseline/MPC")
    control(str(root / "cube_baseline"), kind="cube", steps=50)
    control(str(root / "cube_mpc"), kind="cube", controller="mpc", checkpoint=source, steps=50,
            horizon=3, candidates=8, iterations=1)
    diagnostic = runpy.run_path("scripts/inspect_coverage.py")
    write_json(root / "coverage.json", diagnostic["inspect_coverage"]([0, 1, 2, 3]))
    write_json(root / "independent_null.json", diagnostic["independent_null"]([0, 1, 2, 3]))
    metadata = {"timestamp_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
                "elapsed_seconds": time.perf_counter()-start,
                "platform": platform.platform(), "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in
                             ["torch", "mujoco", "mujoco-warp", "warp-lang", "numpy"]},
                "warp_validation_requested": with_warp,
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in ["src/cube/sim", "src/cube/models", "src/cube/experiments", "configs"]
                                  for p in sorted(Path(folder).glob("*")) if p.suffix in {".py", ".json"}},
                "scope": "Small pipeline checks; no architecture superiority or transfer benefit claim"}
    write_json(root / "run_metadata.json", metadata)
    note("[8/8] Static figures and measured HTML report")
    from cube.reporting import build_report
    build_report(str(root))
    from cube.documentation import build_docs
    build_docs()
    return {"output": str(root), "elapsed_seconds": metadata["elapsed_seconds"],
            "report": "docs/10_results.html", "url": "http://localhost:8000/10_results.html"}
