"""Switchable math backends for wilcat."""

from .backend import (
    math_backend,
    dtype,
    asarray,
    array,
    zeros,
    zeros_like,
    reshape,
    cond,
    inf,
    nan,
)

__all__ = [
    "math_backend",
    "dtype",
    "asarray",
    "array",
    "zeros",
    "zeros_like",
    "reshape",
    "cond",
    "inf",
    "nan",
]
