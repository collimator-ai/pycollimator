from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING, NamedTuple

from enum import IntEnum

import os

import jax
import jax.numpy as jnp

from ..logging import logger
from ..framework import is_event_data

if TYPE_CHECKING:
    from ..framework import (
        SystemBase,
        ContextBase,
        CacheSource,
        CompositeEventCollection,
    )


__all__ = [
    "TimeOrWitnessTriggered",
    "StepResult",
    "GuardIsolationData",
    "SimulatorOptions",
    "SimulatorState",
    "SimulationResults",
    "SolutionData",
    "ResultsOptions",
    "ODESolverOptions",
]


class TimeOrWitnessTriggered(IntEnum):
    kNothingTriggered = 0
    kTimeTriggered = 1
    kWitnessTriggered = 2
    kBothTriggered = 3


class StepResult(IntEnum):
    kReachedPublishTime = 1
    kReachedZeroCrossing = 2
    kReachedUpdateTime = 3
    kTimeHasAdvanced = 4
    kReachedBoundaryTime = 5
    kReachedStepLimit = 6


class GuardIsolationData(NamedTuple):
    triggered: bool
    tf: float
    guards: CompositeEventCollection
    context: ContextBase


@dataclasses.dataclass
class ODESolverOptions:
    rtol: float = 1e-6
    atol: float = 1e-8
    min_step_size: float = None
    max_step_size: float = None
    max_steps: float = None
    method: str = "default"  # Tsit5 (diffrax) or Dopri5 (jax/scipy)
    save_steps: bool = True


# Data structure to mimic diffrax.Solution
class ODESolution(NamedTuple):
    ys: object
    ts: object
    stats: object = None


class SimulatorState(NamedTuple):
    context: ContextBase
    trigger_type: TimeOrWitnessTriggered
    solution: SolutionData
    timed_events: CompositeEventCollection


# Container for options related to the Simulator class.
@dataclasses.dataclass
class SimulatorOptions:
    enable_tracing: bool = os.environ.get("WILDCAT_DISABLE_TRACING", "0") != "1"
    max_major_step_length: float = jnp.inf
    max_major_steps: int = None  # User must set to value for autodiff of lynx.simulate.
    guard_isolation_scale_factor: float = 0.01

    # This option determines whether the simulator saves any data.  If the
    # simulation is initiated from `simulate` this will be set automatically
    # depending on whether `recorded_signals` is provided.  Hence, this
    # should not need to be manually configured.
    save_time_series: bool = False

    # Dictionary of ports (or other cache sources) for which the time series should
    # be recorded. Note that if the simulation is initiated from `simulate` and
    # `recorded_signals` is provided as a kwarg to `simulate`, anything set here
    # will be overridden.  Hence, this should not need to be manually configured.
    recorded_signals: dict[str, CacheSource] = None

    # If the context is not needed for anything, opting to not return it can
    # speed up compilation times.  For instance, typical simulation calls from
    # the UI don't use the context for anything, so model_interface.py will
    # set `return_context=False` for performance.
    return_context: bool = True

    @staticmethod
    def calculate_max_major_steps(system: SystemBase, tspan: tuple[float, float]):
        # For autodiff of lynx.simulate, this path is not possible, JAX
        # throw an error. to work around this, create:
        #   options = SimulatorOptions(max_major_steps=<my value>)
        # outside lynx.simulate, and pass in like this:
        #   lynx.simulate(my_model, options=options)

        # Find the smallest period amongst the periodic events of the system
        if system.periodic_events.has_events:
            min_discrete_step = jax.tree_util.tree_reduce(
                jnp.minimum,
                jax.tree_map(
                    lambda event_data: event_data.period,
                    system.periodic_events,
                    is_leaf=is_event_data,
                ),
            )
            # in this case, we assume that, on average, major steps triggered by
            # zero crossing event, will be as frequent or less frequent than major steps
            # triggered by the smallest discrete period.
            # anything less than 100 is considered inadequate. user can override if they want this.
            max_major_steps = max(100, 2 * int(tspan[1] // min_discrete_step))
            logger.info(
                "max_major_steps=%s based on smallest discrete period=%s",
                max_major_steps,
                min_discrete_step,
            )
        else:
            # in this case we really have no valuable information on which to make an
            # educated guess. who knows how many events might occurr!!!
            # users will have to iterate.
            max_major_steps = 200
            logger.info(
                "max_major_steps=%s by default since no discrete period in system",
                max_major_steps,
            )
        return max_major_steps


# NOTE: Currently unused, but this is where continuous-time interpolation
# should go, along with recorded signals, etc.
@dataclasses.dataclass
class ResultsOptions:
    max_interval_between_samples: float = None
    data_points_min: int = 0
    # TODO: maybe include recorded_signals here?


class SimulationResults(NamedTuple):
    context: ContextBase
    outputs: dict[str, jnp.ndarray]
    data: SolutionData = None

    # These values will hold the "postprocessed" data, which is the results
    # of the simulation after trimming unused buffer entries.  The corresponding
    # `time` and `outputs` in the `data` field hold the actual buffer data and
    # generally should not be accessed directly.  To convert the buffer data,
    # use the `postprocess` method, which will return a new `SimulationResults`
    # object with these fields populated by the postprocessed data.
    time: jnp.ndarray = None
    outputs: dict[str, jnp.ndarray] = None

    def postprocess(self) -> SimulationResults:
        return _postprocess(self)


def _postprocess(raw_results: SimulationResults) -> SimulationResults:
    # Remove NaN data points from the solution buffer (if using adaptive integration)
    time, outputs = raw_results.data.trim()
    return raw_results._replace(time=time, outputs=outputs, data=None)


class SolutionData(NamedTuple):
    outputs: dict[str, jnp.ndarray]
    time: jnp.ndarray
    n_steps: jnp.ndarray  # Number of steps taken by the ODE solver in each interval
    n_major_steps: int = 0  # Number of time intervals (major steps) in the solution

    def trim(self) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Trim unused buffer space from the solution data.

        The raw solution data contains the full 'buffer' of max_steps ODE solver steps and
        max_major_steps time intervals. This function trims the unused buffer space from
        the solution data, the trimmed data.

        Because this returns variable-length arrays depending on the results of the solver
        calls it cannot be called from a JAX jit-compiled function.  Instead, call as part
        of a 'postprocessing' step after simulation is complete.  This is done by default
        if the simulation is invoked via the `simulate` function.
        """
        return _trim(self)


def _trim(solution: SolutionData) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Remove unused entries from the buffer and return flattened arrays.

    See `SolutionData.trim` for more details.
    """
    n_major_steps = solution.n_major_steps
    # n_steps = self.n_steps[:n_major_steps]  # UNUSED
    outputs = {key: y[:n_major_steps] for key, y in solution.outputs.items()}
    time = solution.time[:n_major_steps]

    time = jnp.concatenate(time, axis=0)

    # Adaptive ODE solvers (like diffrax) should return inf for unused buffer entries.
    # Then we can use isfinite to trim the unused buffer space.
    valid_idx = jnp.isfinite(time)
    time = time[valid_idx]

    for key, y in outputs.items():
        y_trim = jnp.concatenate(y, axis=0)
        y_trim = y_trim[valid_idx]
        outputs[key] = y_trim

    return time, outputs
