from typing import Callable, Dict

# Global registry mapping names to builder functions
_DATA_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str) -> Callable:
    """Decorator to register a model builder function under a given name."""

    def decorator(fn: Callable) -> Callable:
        if name in _DATA_REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered.")
        _DATA_REGISTRY[name] = fn
        return fn

    return decorator


def build_dataset(name: str, **kwargs):
    """Instantiate a registered model by name."""
    if name not in _DATA_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return _DATA_REGISTRY[name](**kwargs)


def list_datasets() -> list[str]:
    """Return a sorted list of available models."""
    return sorted(_DATA_REGISTRY)
