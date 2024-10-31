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

import warnings

import pytest
import numpy as np
import jax.numpy as jnp

from collimator.backend import numpy_api as cnp
from collimator.testing import set_backend

try:
    import torch
except ImportError:
    warnings.warn("torch not installed - skipping relevant checks")
    torch = None

float_dtypes = ["float64", "float32", "float16"]
int_dtypes = ["int64", "int32", "int16"]


@pytest.mark.skip(reason="see PR 6523")
def test_switch_backend():
    set_backend("numpy")
    x = cnp.array([0.0, 1.0])
    sin_x = cnp.sin(x)
    assert isinstance(sin_x, np.ndarray)
    assert not isinstance(sin_x, jnp.ndarray)
    assert torch is None or not isinstance(sin_x, torch.Tensor)

    set_backend("jax")
    x = cnp.array([0.0, 1.0])
    sin_x = cnp.sin(x)
    assert not isinstance(sin_x, np.ndarray)
    assert isinstance(sin_x, jnp.ndarray)
    assert torch is None or not isinstance(sin_x, torch.Tensor)

    if torch is not None:
        set_backend("torch")
        x = cnp.array([0.0, 1.0])
        sin_x = cnp.sin(x)
        assert not isinstance(sin_x, np.ndarray)
        assert not isinstance(sin_x, jnp.ndarray)
        assert isinstance(sin_x, torch.Tensor)


@pytest.mark.skip(reason="see PR 6523")
@pytest.mark.parametrize("dtype_str", [*float_dtypes, *int_dtypes])
def test_array(dtype_str):
    x = [1, 2, 3]

    set_backend("numpy")
    dtype = getattr(cnp, dtype_str)
    y = cnp.array(x, dtype=dtype)
    assert isinstance(y, np.ndarray)
    assert y.dtype == dtype
    assert y.shape == (3,)
    assert np.allclose(y, x)

    set_backend("jax")
    dtype = getattr(cnp, dtype_str)
    y = cnp.array(x, dtype=dtype)
    assert isinstance(y, jnp.ndarray)
    assert y.dtype == dtype
    assert y.shape == (3,)
    assert np.allclose(y, x)

    if torch is not None:
        set_backend("torch")
        dtype = getattr(cnp, dtype_str)
        y = cnp.array(x, dtype=dtype)
        assert isinstance(y, torch.Tensor)
        assert y.dtype == dtype
        assert y.shape == (3,)
        assert np.allclose(y, x)


@pytest.mark.skip(reason="see PR 6523")
@pytest.mark.parametrize("dtype_str", [*float_dtypes, *int_dtypes])
def test_zeros_like_vec(dtype_str):
    set_backend("numpy")
    dtype = getattr(cnp, dtype_str)
    x = cnp.array([1, 2, 3], dtype=dtype)
    z = cnp.zeros_like(x)
    assert isinstance(z, np.ndarray)
    assert z.dtype == dtype
    assert z.shape == (3,)
    assert np.all(z == 0.0)

    set_backend("jax")
    dtype = getattr(cnp, dtype_str)
    x = cnp.array([1, 2, 3], dtype=dtype)
    z = cnp.zeros_like(x)
    assert isinstance(z, jnp.ndarray)
    assert z.dtype == dtype
    assert z.shape == (3,)
    assert np.all(z == 0.0)

    if torch is not None:
        set_backend("torch")
        dtype = getattr(cnp, dtype_str)
        x = cnp.array([1, 2, 3], dtype=dtype)
        z = cnp.zeros_like(x)
        assert isinstance(z, torch.Tensor)
        assert z.dtype == dtype
        assert z.shape == (3,)


@pytest.mark.skip(reason="see PR 6523")
@pytest.mark.parametrize("dtype_str", [*float_dtypes, *int_dtypes])
def test_zeros_like_array(dtype_str):
    set_backend("numpy")
    dtype = getattr(cnp, dtype_str)
    x = cnp.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    z = cnp.zeros_like(x)
    assert isinstance(z, np.ndarray)
    assert z.dtype == dtype
    assert z.shape == (2, 3)
    assert np.all(z == 0.0)

    set_backend("jax")
    dtype = getattr(cnp, dtype_str)
    x = cnp.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    z = cnp.zeros_like(x)
    assert isinstance(z, jnp.ndarray)
    assert z.dtype == dtype
    assert z.shape == (2, 3)
    assert np.all(z == 0.0)

    if torch is not None:
        set_backend("torch")
        dtype = getattr(cnp, dtype_str)
        x = cnp.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        z = cnp.zeros_like(x)
        assert isinstance(z, torch.Tensor)
        assert z.dtype == dtype
        assert z.shape == (2, 3)
        assert torch.all(z == 0.0)


@pytest.mark.skip(reason="see PR 6523")
def test_reshape():
    set_backend("numpy")
    x = cnp.array([[1, 2, 3], [4, 5, 6]])
    y = cnp.reshape(x, (3, 2))
    assert isinstance(y, np.ndarray)
    assert y.shape == (3, 2)

    set_backend("jax")
    x = cnp.array([[1, 2, 3], [4, 5, 6]])
    y = cnp.reshape(x, (3, 2))
    assert isinstance(y, jnp.ndarray)
    assert y.shape == (3, 2)

    if torch is not None:
        set_backend("torch")
        x = cnp.array([[1, 2, 3], [4, 5, 6]])
        y = cnp.reshape(x, (3, 2))
        assert isinstance(y, torch.Tensor)
        assert y.shape == (3, 2)
