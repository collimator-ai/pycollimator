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

import numpy as np
import jax
from scipy.integrate._ivp import bdf as sp_bdf
from collimator.backend._jax.bdf import R_matrix, _update_D, MAX_ORDER

jax.config.update("jax_enable_x64", True)


# test the R and D matrix functions by comparing scipy and collimator on same input
def test_RD(verify_R=True, verify_D=True):
    orders = [1, 2, 3, 4, 5]
    factors = [0.1, 0.9, 1.0, 1.1, 2.0]
    for order in orders:
        for factor in factors:
            if verify_R:
                r1 = sp_bdf.compute_R(order, factor)
                r2 = R_matrix(order, factor)
                # scipy creates variable sized arrays for R, but jax doens't allow.
                # in jax, we create padded arrays, so just pad the scipy R arrays for direct comparison.
                pad_int = r2.shape[0] - r1.shape[0]
                r1_filled = np.pad(r1, ((0, pad_int), (0, pad_int)))
                assert np.allclose(r1_filled, r2)

            if verify_D:
                D = np.random.random((8, MAX_ORDER + 3))
                d2 = _update_D(D, order, factor)
                sp_bdf.change_D(D, order, factor)
                assert np.allclose(D, d2)


# the remainder of the bdf codes between collimator and scipy are not
# easily unit tested.
# 1] scipy.solve_bdf_system is sort of like collimator.solve_newton_system, but they require so much
# inputs that need to be setup from running other parts of bdf that i'm sure there is value in testing
# these seperately.
# if it's worth it.
# 2] c.newton_iteration is contained in s._step_impl, and cant really be split out of s. _step_impl
# 3] same goes for c. c.attempt_bdf_step, c._update_difference_matrix, c._update_difference_matrix_order_change
# and thats pretty much all of it.


if __name__ == "__main__":
    test_RD()
