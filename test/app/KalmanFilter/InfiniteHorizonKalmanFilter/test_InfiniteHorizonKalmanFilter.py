#!/bin/env pytest
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

import pytest
import collimator.testing as test
from collimator.testing.markers import skip_if_not_jax

skip_if_not_jax()
pytestmark = pytest.mark.app


def test_InfiniteHorizonKalmanFilter(request):
    test_paths = test.get_paths(request)
    test.run(
        test_paths=test_paths, pytest_request=request, stop_time=1.0, check_only=True
    )
