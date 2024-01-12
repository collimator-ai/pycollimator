"""Aliases for type hints."""
from __future__ import annotations

from typing import Any, Tuple, Union

from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "Array",
    "ArrayLike",
    "ShapeLike",
    "DTypeLike",
    "Scalar",
]

Scalar = ArrayLike  # Specifically, this should be a float, int, or array with shape ()

ShapeLike = Union[Tuple[int, ...], int]
DTypeLike = Any
