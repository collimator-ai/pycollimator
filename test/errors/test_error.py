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

import pytest

from collimator.framework import (
    CollimatorError,
    StaticError,
    ShapeMismatchError,
    BlockParameterError,
)
import collimator.testing as test

this_dir = os.path.dirname(__file__)
output_dir = "test/workdir/errors"


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_add_different_length_vectors(request, backend):
    test.set_backend(backend)
    with pytest.raises(StaticError) as exc:
        test.run(
            request,
            model_json="model_add_different_length_vectors.json",
            stop_time=10.0,
        )
    cause = exc.value.__cause__
    if backend == "jax":
        assert isinstance(cause, TypeError), 'expected a TypeError, got "%s"' % type(
            cause
        )
        assert "got incompatible shapes for broadcasting" in cause.args[0]
    else:
        assert isinstance(cause, ValueError), 'expected a ValueError, got "%s"' % type(
            cause
        )
        assert "operands could not be broadcast together" in cause.args[0]


def test_string_gain(request):
    with pytest.raises(CollimatorError) as e:
        test.run(
            request,
            model_json="model_string_gain.json",
            stop_time=10.0,
        )
    assert e.value.caused_by(TypeError)


def test_name_error(request):
    with pytest.raises(BlockParameterError) as exc:
        test.run(
            request,
            model_json="model_name_error.json",
            stop_time=10.0,
        )
    cause = exc.value.__cause__
    assert isinstance(cause, NameError)
    assert "name 'undefined_name' is not defined" in cause.args[0]


def test_syntax_error(request):
    with pytest.raises(BlockParameterError) as exc:
        test.run(
            request,
            model_json="model_syntax_error.json",
            stop_time=10.0,
        )
    cause = exc.value.__cause__
    assert isinstance(cause, SyntaxError)
    # 3.9: assert "EOL while scanning string literal" in cause.args[0]
    # 3.10: assert "unterminated string literal" in cause.args[0]


def test_integrator_state_wrong_shape(request):
    with pytest.raises(ShapeMismatchError):
        test.run(
            request,
            model_json="model_integrator_state_wrong_shape.json",
            stop_time=10.0,
        )
