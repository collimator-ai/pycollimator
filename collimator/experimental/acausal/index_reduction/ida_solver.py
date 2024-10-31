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
import sympy as sp
from collimator.backend import numpy_api as cnp
from collimator.experimental.acausal.index_reduction.index_reduction import (
    SemiExplicitDAE,
)
from collimator.experimental.acausal.index_reduction.equation_utils import (
    compute_initial_conditions,
)


class IDASolver:
    """
    Solve an 'Acausal system without any causal inputs' with IDA solver.
    The semi-explicit system is converted into an explicit form for IDA.
    ```
    s = [x, y]^T
    F(t, s, sdot) = 0
    ```
    """

    def __init__(
        self,
        sed: SemiExplicitDAE,
        compute_ydot0: bool = False,  # False for index-1 semi-explicit systems
        leaf_backend: str = "jax",
    ):
        self.sed = sed
        self.compute_ydot0 = compute_ydot0
        self.leaf_backend = leaf_backend

        self.n_ode = sed.n_ode
        self.n_alg = sed.n_alg

        self.knowns_symbols, self.knowns_vals = zip(*sed.knowns.items())

    def compute_initial_conditions(self):
        """
        Compute initial conditions for s(t=0)=s0 and sdot(t=0)=sdot0, where s=[x, y]^T.
        """
        sed = self.sed
        X_ic_mapping = compute_initial_conditions(
            sed.t,
            sed.eqs,
            sed.X,
            sed.ics,
            sed.ics_weak,
            sed.knowns,
            verbose=True,
        )

        x_ic = [X_ic_mapping[sed.dae_X_to_X_mapping[var]] for var in sed.x]
        x_dot_ic = [X_ic_mapping[sed.dae_X_to_X_mapping[var]] for var in sed.x_dot]
        y_ic = [X_ic_mapping[sed.dae_X_to_X_mapping[var]] for var in sed.y]

        if self.compute_ydot0:
            ydot0 = self.get_ydot0(x_ic, y_ic)
        else:
            ydot0 = [0.0] * len(y_ic)

        if sed.is_scaled:
            x_ic_scaled = [val / sed.Ss[idx] for idx, val in enumerate(x_ic)]
            x_dot_ic_scaled = [val / sed.Ss[idx] for idx, val in enumerate(x_dot_ic)]
            y_ic_scaled = [
                val / sed.Ss[idx + sed.n_ode] for idx, val in enumerate(y_ic)
            ]
            ydot0_scaled = [
                val / sed.Ss[idx + sed.n_ode] for idx, val in enumerate(ydot0)
            ]

            s0 = x_ic_scaled + y_ic_scaled
            sdot0 = x_dot_ic_scaled + list(ydot0_scaled)
        else:
            s0 = x_ic + y_ic
            sdot0 = x_dot_ic + ydot0

        return s0, sdot0

    def get_ydot0(self, x0, y0):
        """
        Compute ydot0. This doesn't need to be called for semi-explicit index-1 systems
        as the solution won't be affected by the choice of ydot0.
        Needs modification for higher index (>1) systems.
        """
        sed = self.sed
        args = (sed.t, sed.x, sed.y, self.knowns_symbols)

        self.gy = sp.Matrix(sed.g).jacobian(sed.y)
        self.gx = sp.Matrix(sed.g).jacobian(sed.x)

        gy = sp.lambdify(
            args,
            self.gy,
            modules=[self.leaf_backend, {"cnp": cnp}],
        )
        gx = sp.lambdify(
            args,
            self.gx,
            modules=[self.leaf_backend, {"cnp": cnp}],
        )

        ydot0 = np.linalg.solve(
            gy(0.0, x0, y0, self.knowns_vals),
            -gx(0.0, x0, y0, self.knowns_vals) @ np.array(x0),
        )
        return ydot0

    def create_residual_and_jacobian(self):
        """
        For the system `F(t, s, sdot) = 0`, create functions for the the residual `F`
        and its jacobian w.r.t `s` and `sdot` for the IDA solver.
        """
        sed = self.sed
        self.s = sed.x + sed.y
        self.sdot = sed.x_dot + [sp.Derivative(y, sed.t) for y in sed.y]

        self.F = [xdot - fx for xdot, fx in zip(sed.x_dot, sed.f)] + sed.g

        self.dF_ds = sp.Matrix(self.F).jacobian(self.s)
        self.dF_dsdot = sp.Matrix(self.F).jacobian(self.sdot)

        self.lambda_args = (sed.t, self.s, self.sdot, self.knowns_symbols)

        F = sp.lambdify(
            self.lambda_args,
            self.F,
            modules=[self.leaf_backend, {"cnp": cnp}],
        )

        dF_ds = sp.lambdify(
            self.lambda_args,
            self.dF_ds,
            modules=[self.leaf_backend, {"cnp": cnp}],
        )

        dF_dsdot = sp.lambdify(
            self.lambda_args,
            self.dF_dsdot,
            modules=[self.leaf_backend, {"cnp": cnp}],
        )

        return F, dF_ds, dF_dsdot

    def solve(
        self,
        sim_time: float = 1.0,
        dt: float = 0.1,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        first_step_size: float = 1e-18,
        exclude_algvar_from_error: bool = False,
        max_steps: int = 500,
    ):
        from scikits.odes import dae

        s0, sdot0 = self.compute_initial_conditions()
        F, dF_ds, dF_dsdot = self.create_residual_and_jacobian()

        def dae_residual(t, s, sdot, result):
            result[:] = F(t, s, sdot, self.knowns_vals)
            return 0

        def dae_jacobian(t, s, sdot, residual, cj, result):
            result[:, :] = dF_ds(t, s, sdot, self.knowns_vals) + cj * dF_dsdot(
                t, s, sdot, self.knowns_vals
            )
            return 0

        solver = dae(
            "ida",
            dae_residual,
            jacfn=dae_jacobian,
            first_step_size=first_step_size,
            rtol=rtol,
            atol=atol,
            algebraic_vars_idx=list(range(self.n_ode, self.n_ode + self.n_alg)),
            exclude_algvar_from_error=exclude_algvar_from_error,
            max_steps=max_steps,
            old_api=False,
        )

        time = 0.0
        solver.init_step(time, s0, sdot0)

        t_sol = [time]
        s_sol = [s0]
        while True:
            time += dt
            solution = solver.step(time)
            if solution.errors.t:
                print(f"Error: {solution.message} at time {solution.errors.t}")
                break
            t_sol.append(solution.values.t)
            s_sol.append(solution.values.y)
            if time >= sim_time:
                break

        t_sol_arr = np.array(t_sol)
        s_sol_arr = np.array(s_sol)
        print(f"{self.sed.is_scaled=} {'&&'*100}")
        if self.sed.is_scaled:
            s_sol_arr = s_sol_arr * self.sed.Ss
        return t_sol_arr, s_sol_arr
