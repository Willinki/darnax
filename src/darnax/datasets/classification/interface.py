from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

    import jax

#: Type alias for rescaling modes available to all datasets.
RescalingMode = Literal["default", "null", "divide255", "standardize"]


class ClassificationDataset(ABC):
    """Abstract base class for datasets compatible with darnax trainers.

    Datasets must implement train/test iteration and provide metadata via `spec()`.
    Validation split is optional.

    Required Methods
    ----------------
    - build(key) : Load and preprocess data
    - __iter__() : Training batch iterator
    - iter_test() : Test batch iterator
    - __len__() : Number of training batches
    - spec() : Dataset metadata and structure

    Optional Methods
    ----------------
    - iter_valid() : Validation batch iterator (default raises NotImplementedError)

    """

    @abstractmethod
    def build(self, key: jax.Array) -> jax.Array:
        """Load, preprocess, and prepare dataset splits."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over training batches."""
        pass

    @abstractmethod
    def iter_test(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over test batches."""
        pass

    def iter_valid(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over validation batches (optional)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide a validation split. "
            "Override `iter_valid()` to provide validation data."
        )

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of training batches."""
        pass

    @abstractmethod
    def spec(self) -> dict[str, Any]:
        """Return dataset specification with metadata."""
        pass

    def _apply_rescaling(self, x: jax.Array) -> jax.Array:
        """Apply rescaling according to rescaling mode.

        Modes
        -----
        - "default": Use dataset's DEFAULT_RESCALING class property.
        - "null": No rescaling applied.
        - "divide255": Divide by 255 (for uint8 images, maps [0,255] to [0,1]).
        - "standardize": Zero mean, unit variance.
        """
        mode = self.rescaling
        if mode == "default":
            mode = self.DEFAULT_RESCALING
            if mode is None:
                return x
        if mode == "null":
            return x
        elif mode == "divide255":
            return x / 255.0
        elif mode == "standardize":
            return (x - x.mean()) / x.std()
        return x
