import os

import jax

from . import logging
from .framework import (
    LeafSystem,
    DiagramBuilder,
)
from .math_backend import math_backend

from .simulation import Simulator, simulate, odeint, ODESolver
from .cli import load_model, loads_model

jax.config.update("jax_enable_x64", True)  # Turn off for float32 (limits accuracy)

_log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.set_log_level(_log_level)

_per_package_log_levels = os.environ.get("LOG_LEVELS", None)
if _per_package_log_levels is not None:
    _per_package_log_levels = _per_package_log_levels.split(",")
    _per_package_log_levels = [level.split(":") for level in _per_package_log_levels]
    for pkg, level in _per_package_log_levels:
        logging.set_log_level(level, pkg=pkg)

logging.set_log_handlers()

set_backend = math_backend.set_backend

__all__ = [
    "load_model",
    "loads_model",
    "LeafSystem",
    "DiagramBuilder",
    "Simulator",
    "simulate",
    "odeint",
    "ODESolver",
    "math_backend",
    "set_backend",
]
