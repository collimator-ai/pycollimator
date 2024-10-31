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

import os

from collimator.backend import numpy_api, DEFAULT_BACKEND, REQUESTED_BACKEND
from collimator.framework.cache import BasicOutputCache
from collimator.lazy_loader import LazyLoader

pytest = LazyLoader("pytest", globals(), "pytest")


# NOTE: skipping all acausal tests for now:
# ValueError: The solver ScipySolver does not support systems with non-trivial mass matrices.
# Use a different solver (currently only the JAX-backend BDF solver is compatible with mass-matrix ODEs).


def requires_jax(xfail=False):
    """Custom decorator for tests that must run with JAX backend"""

    def decorator(func):
        if DEFAULT_BACKEND == "jax":
            return func
        if not xfail:
            return pytest.mark.skip(reason="Test requires JAX backend")(func)
        return pytest.mark.xfail(reason="Test fails with non-JAX backend")(func)

    return decorator


def skip_if_not_jax():
    """Skip test if not using JAX backend"""

    if DEFAULT_BACKEND == "jax":
        return

    pytest.skip(reason="Test requires JAX backend", allow_module_level=True)


def set_backend(name):
    # If we explicitly requested numpy, skip testing other backends.
    # This is to support CI checks with COLLIMATOR_BACKEND=numpy,
    # eg. for pyodide (wasm with jaxlite) or testing pure numpy.

    if (
        pytest is not None
        and os.environ.get("PYTEST_CURRENT_TEST") is not None
        and REQUESTED_BACKEND is not None
        and name != REQUESTED_BACKEND
    ):
        pytest.skip(f"Requested backend {name} not available for test")

    numpy_api.set_backend(name)

    # Invalidate all caches and enable for numpy.
    BasicOutputCache.activate(name)
