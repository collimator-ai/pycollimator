from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from .types import SolutionData
from ..framework import ContextBase

if TYPE_CHECKING:
    from .ode_solver import ODESolution
    from ..framework import CacheSource


__all__ = [
    "make_empty_solution",
    "update_sim_solution",
]


def make_empty_solution(
    context, recorded_signals, max_steps=16**3, max_major_steps=100
) -> SolutionData:
    # For each source in "recorded_signals", determine what the signal data type is
    # and create an empty vector that can hold enough data to max out the simulation
    # buffer.

    def _expand_template(source):
        # Determine the data type of the signal (shape and dtype)
        x = source.eval(context)
        if jnp.isscalar(x):
            x = jnp.asarray(x)
        if not isinstance(x, jnp.ndarray):
            return None
        # Create a buffer that can hold the maximum number of (major, minor) steps
        return jnp.zeros((max_major_steps, max_steps + 1, *x.shape), dtype=x.dtype)

    signals = {
        key: _expand_template(source) for key, source in recorded_signals.items()
    }
    # The time vector is used to determine the number of steps taken by the ODE solver
    # since diffrax will return inf for unused buffer entries. Then we can use isfinite
    # to trim the unused buffer space.  For this reason, initialize to inf rather
    # than zero.
    times = jnp.full((max_major_steps, max_steps + 1), jnp.inf)
    n_steps = jnp.zeros((max_major_steps,), dtype=jnp.int32)
    return SolutionData(signals, times, n_steps)


def _get_at_index(states, index):
    """Isolate the state values at a given time step"""
    return jax.tree_map(lambda x: x[index], states)


def _eval_source(source, context, xc, t, index):
    """Evaluate a cache source at a given time step"""
    context = context.with_continuous_state(_get_at_index(xc, index))
    context = context.with_time(t[index])
    # logger.debug(f"Eval cache source with state {context.state}")
    result = source.eval(context)
    return result


def update_sim_solution(
    context: ContextBase,
    solution: SolutionData,
    source_dict: dict[str, CacheSource],
    ode_sol: ODESolution,
) -> SolutionData:
    """Update the simulation solution with the results of a simulation step."""

    # Index of the current major step in the solution data.
    index = solution.n_major_steps

    if ode_sol is not None:
        # Reconstruct the signals at all ODE solver steps

        # The second entries in `ode_sol` contain all steps (first is just the end result)
        ys, ts = ode_sol.ys[1], ode_sol.ts[1]
        n_steps = ode_sol.stats["num_accepted_steps"]

        def _scan_fun(port, carry, i):
            return carry, _eval_source(port, *carry, i)

        outputs = solution.outputs.copy()
        carry = (context, ys, ts)
        for key, source in source_dict.items():
            # Reconstruct the signal at all ODE solver steps
            _, y = lax.scan(partial(_scan_fun, source), carry, jnp.arange(len(ts)))
            outputs[key] = outputs[key].at[index].set(y)

        time = solution.time.at[index].set(ts)

    else:
        # Pure discrete system: no ODE solver steps

        ys = context.continuous_state
        n_steps = 1

        outputs = solution.outputs.copy()
        for key, source in source_dict.items():
            # In this case we only need to get the signal at the current step,
            # since there are no intermediate steps from the ODE solver.
            y = source.eval(context)
            outputs[key] = outputs[key].at[index, 0].set(y)

        # Set the first entry of the time vector to the current time.
        # The rest will still be inf, indicating unused buffer entries.
        time = solution.time.at[index, 0].set(context.time)

    solution = solution._replace(
        outputs=outputs,
        time=time,
        n_steps=solution.n_steps.at[index].set(n_steps),
        n_major_steps=index + 1,
    )

    return solution
