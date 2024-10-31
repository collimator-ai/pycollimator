# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

from . import _init  # noqa: F401
from .framework import (
    LeafSystem,
    DiagramBuilder,
    Parameter,
    parameters,
    ports,
)
from .backend import dispatcher as backend

from .library.nmpc import trajopt
from .library.linear_system import linearize

from .simulation import (
    Simulator,
    simulate,
    estimate_max_major_steps,
    ODESolver,
    ODESolverOptions,
    SimulatorOptions,
)
from .cli import load_model, load_model_from_dir
from .version import __version__

set_backend = backend.set_backend

__all__ = [
    "__version__",
    "load_model",
    "load_model_from_dir",
    "linearize",
    "LeafSystem",
    "DiagramBuilder",
    "Simulator",
    "SimulatorOptions",
    "simulate",
    "trajopt",
    "estimate_max_major_steps",
    "ODESolver",
    "SimulatorOptions",
    "ODESolverOptions",
    "backend",
    "set_backend",
    "Parameter",
    "parameters",
    "ports",
]
