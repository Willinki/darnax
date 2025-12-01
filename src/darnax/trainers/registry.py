from darnax.trainers.interface import AbstractOrchestrator
from darnax.trainers.dynamical import DynamicalTrainer
from darnax.trainers.hebbian_contrastive import ContrastiveHebbianTrainer
from darnax.trainers.alternate import AltTrainer
from local_exp.trainers.dynamical_v2 import DynamicalTrainerV2
from typing import Callable, Dict

_TRAINER_REGISTRY: Dict[str, Callable] = {
    "dynamical": DynamicalTrainer,
    "contrastive": ContrastiveHebbianTrainer,
    "alternative": AltTrainer,
    "dynamical_v2": DynamicalTrainerV2,
}


def build_trainer(name: str, **kwargs):
    """Instantiate a registered model by name."""
    if name not in _TRAINER_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return _TRAINER_REGISTRY[name](**kwargs)


def list_trainers() -> list[str]:
    """Return a sorted list of available models."""
    return sorted(_TRAINER_REGISTRY)
