"""Public experiment commands used by the docker-compose CLI."""
from .data import generate_dataset
from .learning import evaluate, load_checkpoint, train
from .control import ant_gait_action, control, shooting_mpc
from .protocols import coupling, run_suite, transfer, validate

__all__ = ["generate_dataset", "train", "evaluate", "load_checkpoint", "validate", "coupling",
           "control", "run_suite", "transfer", "ant_gait_action", "shooting_mpc"]
