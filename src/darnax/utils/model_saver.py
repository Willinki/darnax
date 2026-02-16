"""Save/load lightweight inference snapshots of a trainer's orchestrator parameters.

This module serializes only selected leaves (default: inexact arrays) from
`trainer.orchestrator`, plus optional metadata and simple metric summaries, into
a compact, inference-friendly snapshot. It also provides a loader to restore
those parameters into a compatible template orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union
import time
import hashlib
import numpy as np

import equinox as eqx
import jax

PathLike = Union[str, Path]


def _resolve_file(path: PathLike, *, filename: str) -> Path:
    """
    If `path` looks like a file (has a suffix), return it.
    Otherwise treat it as a directory and append `filename`.
    """
    p = Path(path)
    if not filename:
        raise ValueError("`filename` must be a non-empty string.")
    return p if p.suffix else (p / filename)


def _pytree_signature(
    pytree: Any,
    *,
    filter_spec: Callable[[Any], bool],
) -> dict[str, Any]:
    """
    Build a lightweight, JSON-serializable signature of the filtered leaves:
    a list of (shape, dtype) for each saved leaf in traversal order.
    This is for debugging / mismatch detection (not a guarantee of compatibility).
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(pytree, filter_spec))
    sig = []
    for leaf in leaves:
        # For non-selected leaves, eqx.filter returns None; ignore them.
        if leaf is None:
            continue
        # JAX arrays / numpy arrays both have .shape and .dtype
        sig.append({"shape": tuple(int(x) for x in leaf.shape), "dtype": str(leaf.dtype)})
    return {"n_leaves": len(sig), "leaves": sig}


def orchestrator_saver(
    trainer: Any,
    path: PathLike,
    *,
    filename: str = "orchestrator.eqx",
    filter_spec: Callable[[Any], bool] = eqx.is_inexact_array,
    metadata: Optional[Mapping[str, Any]] = None,
    metadata_filename: str = "metadata.json",
    include_signature: bool = True,
) -> Path:
    """
    Save an inference snapshot of the trained model: ONLY the inexact-array leaves
    of `trainer.orchestrator` (typically float/complex parameters).

    This does NOT save:
      - architecture / static fields
      - optimizer state
      - RNG keys
      - dataset
      - trainer internals other than orchestrator parameters

    Loading requires:
      - you rebuild a template trainer with an orchestrator of identical structure
        and parameter shapes, then load into it.

    Args:
        trainer: Any object with attribute `.orchestrator` which is an eqx.Module/PyTree.
        path: Directory or file path. If directory, writes `filename` inside it.
        filename: Params filename used when `path` is a directory.
        filter_spec: Leaf predicate to select what to save (default: eqx.is_inexact_array).
        metadata: Optional JSON-serializable metadata (e.g., your PARAMS dict).
        metadata_filename: JSON filename written next to the params file.
        include_signature: If True, add a (shape,dtype) signature to metadata.

    Returns:
        The resolved params file path written.
    """
    
    if not hasattr(trainer, "orchestrator"):
        raise AttributeError("`trainer` must have an `.orchestrator` attribute.")
    if not callable(filter_spec):
        raise TypeError("`filter_spec` must be callable.")

    file_path = _resolve_file(path, filename=filename)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory: {file_path.parent}") from e

    try:
        params = eqx.filter(trainer.orchestrator, filter_spec)
        eqx.tree_serialise_leaves(str(file_path), params)
    except Exception as e:
        raise RuntimeError(f"Failed to serialize parameters to {file_path}") from e

    if metadata is not None or include_signature:
        try:
            meta: dict[str, Any] = dict(metadata) if metadata is not None else {}
            if include_signature:
                meta.setdefault(
                    "orchestrator_signature",
                    _pytree_signature(trainer.orchestrator, filter_spec=filter_spec),
                )
            meta_path = file_path.parent / metadata_filename
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        except (TypeError, ValueError) as e:
            raise ValueError("Metadata must be JSON-serializable.") from e
        except OSError as e:
            raise OSError(f"Failed to write metadata to {meta_path}") from e

    return file_path


def _make_run_id(cfg: dict[str, Any]) -> str:
    try:
        cfg_bytes = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise ValueError("`cfg` must be JSON-serializable (or convertible via `default=str`).") from e
    return hashlib.sha256(cfg_bytes).hexdigest()[:12]


def _metrics_stats(history: Mapping[str, list[float]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for k, v in history.items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.size == 0:
            stats[k] = {"mean": None, "std": None, "min": None, "max": None, "last": None}
        else:
            stats[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "last": float(arr[-1]),
            }
    return stats


def save_inference_orchestrator(
    trainer: Any,
    cfg: dict[str, Any],
    out_dir: PathLike = "runs",
    *,
    filename: str = "orchestrator.eqx",
    filter_spec: Callable[[Any], bool] = eqx.is_inexact_array,
    metadata: Optional[Mapping[str, Any]] = None,
    metadata_filename: str = "metadata.json",
    include_signature: bool = True,
    metrics: Optional[Mapping[str, list[float]]] = None,
) -> Path:
    """
    Save orchestrator into a timestamped run folder like saver.py:
      runs/YYYYMMDD-HHMMSS_run-<hash>/{orchestrator.eqx, metadata.json, metrics.npz}
    """
    
    if not isinstance(cfg, dict):
        raise TypeError("`cfg` must be a dict.")
    run_id = _make_run_id(cfg)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}_run-{run_id}"
    run_path = Path(out_dir) / run_name
    try:
        run_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create run directory: {run_path}") from e

    meta = dict(metadata) if metadata is not None else {}
    meta.update({"run_id": run_id, "timestamp": ts, "config": cfg})

    if metrics is not None:
        try:
            meta["metrics"] = metrics
            meta["statistics"] = _metrics_stats(metrics)
            np.savez(
                run_path / "metrics.npz",
                **{k: np.asarray(v, dtype=np.float32) for k, v in metrics.items()},
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save metrics to {run_path / 'metrics.npz'}") from e

    return orchestrator_saver(
        trainer=trainer,
        path=run_path,
        filename=filename,
        filter_spec=filter_spec,
        metadata=meta,
        metadata_filename=metadata_filename,
        include_signature=include_signature,
    )


def load_inference_orchestrator(
    trainer_template: Any,
    path: PathLike,
    *,
    filename: str = "orchestrator.eqx",
    filter_spec: Callable[[Any], bool] = eqx.is_inexact_array,
) -> Any:
    """
    Load an inference snapshot into `trainer_template.orchestrator` and return a trainer
    with the loaded orchestrator.

    IMPORTANT:
      - `trainer_template.orchestrator` must have identical filtered-leaf structure and
        shapes as the saved model.
      - This function does not reconstruct architecture; it overwrites parameters only.

    Args:
        trainer_template: A trainer object whose `.orchestrator` matches the saved one.
        path: Directory or file path. If directory, reads `filename` inside it.
        filename: Params filename used when `path` is a directory.
        filter_spec: Leaf predicate used at save time (must match).

    Returns:
        The same `trainer_template` object, with `.orchestrator` replaced by loaded params.
        (Mutates in place.)
    """
    
    if not hasattr(trainer_template, "orchestrator"):
        raise AttributeError("`trainer_template` must have an `.orchestrator` attribute.")
    if not callable(filter_spec):
        raise TypeError("`filter_spec` must be callable.")

    file_path = _resolve_file(path, filename=filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Params file not found: {file_path}")

    try:
        params_template, static = eqx.partition(trainer_template.orchestrator, filter_spec)
        loaded_params = eqx.tree_deserialise_leaves(str(file_path), params_template)
        trainer_template.orchestrator = eqx.combine(loaded_params, static)
    except Exception as e:
        raise RuntimeError(f"Failed to load parameters from {file_path}") from e

    return trainer_template
