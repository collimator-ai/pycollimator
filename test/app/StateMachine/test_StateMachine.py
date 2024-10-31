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
import matplotlib.pyplot as plt
import numpy as np

pytestmark = pytest.mark.app


def test_StateMachine(request, show_plot=False):
    test.run(pytest_request=request, stop_time=10.0, check_only=True)


@pytest.mark.xfail(
    reason="agnostic state machine and relay use events whihc fail during init"
)
def test_smach_init(request, show_plot=False):
    r = test.run(pytest_request=request, stop_time=10.0, model_json="smach_init.json")

    print(r)

    sol = np.where(r["time"] < 5.0, 5.0, 10.0)
    sol[0] = 10.0

    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(r["time"], r["sm_agnostic.out_0"], label="sm_agnostic")
        ax.plot(r["time"], r["Relay_0.out_0"], label="Relay_0")
        ax.plot(r["time"], r["sm_discrete.out_0"], label="sm_discrete")
        ax.plot(r["time"], sol, label="sol")
        ax.legend()
        plt.show()

    assert np.allclose(sol, r["sm_agnostic.out_0"])
    assert np.allclose(sol, r["Relay_0.out_0"])
    assert np.allclose(sol, r["sm_discrete.out_0"])
