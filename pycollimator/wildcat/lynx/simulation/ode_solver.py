from __future__ import annotations
import dataclasses
from functools import partial
from typing import TYPE_CHECKING, Tuple, ClassVar

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

import numpy as np
from scipy.integrate import solve_ivp
from jax.experimental.ode import odeint as jax_odeint
from .rk4 import odeint as rk4_odeint

import diffrax
from diffrax import (
    ODETerm,
    SaveAt,
    SubSaveAt,
    diffeqsolve,
    PIDController,
)

from jax import tree_util

from .types import ODESolution, ODESolverOptions
from ..math_backend.typing import Array

if TYPE_CHECKING:
    from ..framework import SystemBase, ContextBase, State

__all__ = [
    "odeint",
    "ODESolver",
]


def odeint(
    system,
    context,
    t_span,
    t_eval=None,
    return_context=False,
    ode_options: ODESolverOptions = None,
):
    """Interface to diffrax ODE solvers.

    This is not directly used in the main simulation loop, but is intended to
    be a SciPy-like interface for development/testing/etc in continuous-time-only
    systems.
    """
    if ode_options is None:
        ode_options = ODESolverOptions()
    ode_options.save_steps = t_eval is not None

    solve = ODESolver(system, options=ode_options)
    sol, context = solve(context, t_span, t_eval=t_eval)

    if t_eval is None:
        ys = sol.ys[0]
    else:
        ys = sol.ys[1]

    return ys


@dataclasses.dataclass
class ODESolverBase:
    """Common interface for defining ODE solvers.

    The ODE solving function has signature:
    ```
    def __call__(
        context: ContextBase, t_span: Tuple[float, float], t_eval: Array = None
    ): -> (sol: ODESolution, context: ContextBase)
    ```
    and can be used for instance to advance continuous time during hybrid
    simulation.
    """

    system: SystemBase
    rtol: float = 1e-6
    atol: float = 1e-8
    max_steps: int = None
    max_step_size: float = None
    min_step_size: float = None
    method: str = "default"
    save_steps: bool = True

    DEFAULT_MAX_STEPS: ClassVar[int] = 100

    @staticmethod
    def ode_rhs(system, t, y, context):
        context = context.with_time(t)
        context = context.with_continuous_state(y)
        xcdot = system.eval_time_derivatives(context)
        return xcdot

    def _finalize(self):
        pass

    def __post_init__(self):
        if self.max_steps is None:
            self.max_steps = self.DEFAULT_MAX_STEPS
        self._finalize()

    def _solve(
        self, context: ContextBase, t_span: Tuple[float, float], t_eval: Array = None
    ) -> Tuple[ODESolution, ContextBase]:
        raise NotImplementedError(
            "ODESolver._solve must be implemented or created by _finalize"
        )

    def __call__(
        self,
        context: ContextBase,
        t_span: Tuple[float, float],
        t_eval: Array = None,
        x0: State = None,
    ) -> Tuple[ODESolution, ContextBase]:
        t0, tf = t_span
        context = context.with_time(t0)
        if x0 is not None:
            context = context.with_state(x0)

        # this is needed in the case of system without continuous states,
        # but with unrestricted events, i.e. events that may only be based
        # on some function which depends only on time.
        if context.num_continuous_states == 0:
            return None, context.with_time(tf)

        ode_sol, context = self._solve(context, t_span, t_eval=t_eval)

        # The first entry of the ode_sol is the final state (second is all solver steps)
        xc = ode_sol.ys[0]

        # logger.debug(f"Integrated from t={context.time} to t={tf}: result={xc}")
        context = context.with_continuous_state(xc)
        context = context.with_time(tf)
        return ode_sol, context


class DiffraxSolver(ODESolverBase):
    supported_methods = {
        "default": diffrax.Tsit5,
        "Euler": diffrax.Euler,
        "Tsit5": diffrax.Tsit5,
        "Heun": diffrax.Heun,
        "Midpoint": diffrax.Midpoint,
        "Ralston": diffrax.Ralston,
        "Bosh3": diffrax.Bosh3,
        "Dopri5": diffrax.Dopri5,
        "Dopri8": diffrax.Dopri8,
        "ImplicitEuler": diffrax.ImplicitEuler,
        "Kvaerno3": diffrax.Kvaerno3,
        "Kvaerno4": diffrax.Kvaerno4,
        "Kvaerno5": diffrax.Kvaerno5,
    }

    DEFAULT_MAX_STEPS: ClassVar[int] = 16**3

    def _finalize(self):
        """Create a wrapper for diffrax.diffeqsolve for ODE solving.

        This is the preferred method for ODE solving, but it requires that the system
        time derivatives are traceable by JAX.
        """
        term = ODETerm(partial(self.ode_rhs, self.system))

        stepsize_controller = PIDController(
            rtol=self.rtol,
            atol=self.atol,
            dtmin=self.min_step_size,
            dtmax=self.max_step_size,
        )

        try:
            solver = self.supported_methods[self.method]()
        except KeyError:
            raise ValueError(
                f"Invalid method '{self.method}' for JAX ODE solver. Must be one of "
                f"{list(self.supported_methods.keys())}"
            )

        def _ode_solve(
            context: ContextBase, t_span: Tuple[float, float], t_eval: Array = None
        ) -> Tuple[ODESolution, ContextBase]:
            xc0 = context.continuous_state

            if t_eval is None:
                if self.save_steps:
                    # Save both the final state and all the intermediate solver steps
                    saveat = SaveAt(
                        subs=[
                            SubSaveAt(t0=False, t1=True),
                            SubSaveAt(t0=True, steps=True),
                        ]
                    )
                else:
                    # Only save the final state
                    saveat = SaveAt(t0=False, t1=True)
            else:
                # Save both the final state and the requested time points
                saveat = SaveAt(
                    subs=[
                        SubSaveAt(t0=False, t1=True),
                        SubSaveAt(t0=False, ts=t_eval),
                    ]
                )

            sol = diffeqsolve(
                term,
                solver,
                t_span[0],
                t_span[1],
                None,
                xc0,
                context,
                stepsize_controller=stepsize_controller,
                saveat=saveat,
                max_steps=self.max_steps,
            )

            if self.save_steps:
                num_accepted_steps = sol.stats["num_accepted_steps"]
                xf = sol.ys[0]
                xs = sol.ys[1]
                tf = sol.ts[0]
                # The solution handling will record the final value separately,
                # so set the final time to infinity to avoid duplicate entries.
                # This will get trimmed out of the solution later
                # (see SolutionData.trim)
                ts = sol.ts[1].at[num_accepted_steps].set(sol.ts[1][-1])
            else:
                xf = sol.ys
                xs = None
                tf = sol.ts
                ts = None

            sol_out = ODESolution((xf, xs), (tf, ts), sol.stats)
            return sol_out, context

        self._solve = jax.jit(_ode_solve)


class JaxSolver(ODESolverBase):
    # NOTE: This is currently broken with autodiff - throws an error
    # ValueError: dtype=dtype([('float0', 'V')]) is not a valid dtype for JAX type promotion.
    # from somewhere jax.experimental.ode._odeint_rev
    supported_methods = {
        "Dopri5": jax_odeint,
        "RK4": rk4_odeint,
    }

    def _finalize(self):
        """Create a wrapper for jax.experimental.odeint for ODE solving.

        This is the preferred method for ODE solving, but it requires that the system
        time derivatives are traceable by JAX.
        """
        # Extract flattened symbolic variables from the context along with information
        # to recreate the tree structure from the solution data.

        options = {
            "rtol": self.rtol,
            "atol": self.atol,
            "hmax": self.max_step_size or np.inf,
        }

        try:
            odeint = self.supported_methods[self.method]
        except KeyError:
            raise ValueError(
                f"Invalid method '{self.method}' for JAX ODE solver. Must be one of "
                f"{list(self.supported_methods.keys())}"
            )

        def func(y, t, context):
            return self.ode_rhs(self.system, t, y, context)

        def _ode_solve(
            context: ContextBase, t_span: Tuple[float, float], t_eval: Array = None
        ) -> Tuple[ODESolution, ContextBase]:
            xc0 = context.continuous_state

            if t_eval is None:
                if self.save_steps:
                    t_eval = jnp.linspace(t_span[0], t_span[1], self.max_steps + 1)
                else:
                    # Only save the final state
                    t_eval = jnp.asarray(t_span)
            else:
                t_eval = jnp.asarray(t_eval)

            ys = odeint(
                func,
                xc0,
                t_eval,
                context,
                **options,
            )

            # Extract the final time/state
            num_accepted_steps = len(t_eval) - 1
            xf = tree_util.tree_map(lambda x: x[num_accepted_steps], ys)
            tf = t_eval[num_accepted_steps]
            if self.save_steps:
                xs = ys
                ts = t_eval
            else:
                xs = None
                ts = None

            stats = {"num_accepted_steps": num_accepted_steps}

            sol_out = ODESolution((xf, xs), (tf, ts), stats)
            return sol_out, context

        self._solve = jax.jit(_ode_solve)


class ScipySolver(ODESolverBase):
    supported_methods = {
        "default": "RK45",
        "RK45": "RK45",
        "RK23": "RK23",
        "DOP853": "DOP853",
        "Radau": "Radau",
        "BDF": "BDF",
        "LSODA": "LSODA",
    }

    def make_ravel(self, pytree):
        x, unravel = ravel_pytree(pytree)

        def ravel(x):
            return np.hstack(tree_util.tree_leaves(x))

        return x, ravel, unravel

    def _finalize(self):
        """Create a wrapper for scipy.integrate.solve_ivp for ODE solving

        This can be used in cases where diffrax cannot - specifically when the system
        time derivatives are not traceable by JAX.
        """
        try:
            method = self.supported_methods[self.method]
        except KeyError:
            raise ValueError(
                f"Invalid method '{self.method}' for SciPy ODE solver. Must be one of "
                f"{list(self.supported_methods.keys())}"
            )

        self.options = {
            "method": method,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step": self.max_step_size or np.inf,
        }

        if self.method == "LSODA":
            self.options["min_step"] = self.min_step_size or 0.0

    def _solve(
        self, context: ContextBase, t_span: Tuple[float, float], t_eval: Array = None
    ) -> Tuple[ODESolution, ContextBase]:
        # Extract flattened symbolic variables from the context along with information
        # to recreate the tree structure from the solution data.
        xc0, ravel, unravel = self.make_ravel(context.continuous_state)

        def f(t, y, context):
            xc = unravel(y)
            xcdot = self.ode_rhs(self.system, t, xc, context)
            return ravel(xcdot)

        sol = solve_ivp(f, t_span, xc0, t_eval=t_eval, args=(context,), **self.options)
        num_accepted_steps = len(sol.t) - 1

        # The default diffrax solver uses a fixed "buffer" size because JAX arrays
        # are immutable and cannot vary in shape. To simulate this, pad the solution
        # with NaNs to the maximum number of steps.
        if len(sol.t) > self.max_steps:
            raise RuntimeError(
                "TODO: Support variable solution length when not using JAX"
            )
        else:
            sol.t = np.append(sol.t, np.full((self.max_steps - len(sol.t),), np.nan))
            sol.y = np.append(
                sol.y,
                np.full((sol.y.shape[0], self.max_steps - sol.y.shape[1]), np.nan),
                axis=1,
            )

        # Restore the pytree structure
        xs = jax.vmap(unravel)(sol.y.T)
        ts = sol.t

        # Extract the final time/state
        xf = tree_util.tree_map(lambda x: x[num_accepted_steps], xs)
        tf = ts[num_accepted_steps]

        # Diffrax will return a tuple where the first element is the final state and
        # the second element is a list of all the intermediate steps.
        stats = {"num_accepted_steps": num_accepted_steps}

        sol_out = ODESolution((xf, xs), (tf, ts), stats)
        return sol_out, context


def ODESolver(system: SystemBase, options: ODESolverOptions = None) -> ODESolverBase:
    """Create an ODE solver used to advance continuous time in hybrid simulation."""
    if options is None:
        options = ODESolverOptions()
    options = dataclasses.asdict(options)

    if system.enable_trace_time_derivatives:
        if options["method"] in JaxSolver.supported_methods:
            return JaxSolver(system, **options)
        return DiffraxSolver(system, **options)
    else:
        return ScipySolver(system, **options)
