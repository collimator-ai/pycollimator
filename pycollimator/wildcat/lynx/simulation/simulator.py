from __future__ import annotations
from functools import partial
import dataclasses
from typing import TYPE_CHECKING, SupportsIndex

import jax
import jax.numpy as jnp
from jax import lax

from ..logging import logger
from ..profiling import Profiler

from .types import (
    TimeOrWitnessTriggered,
    StepResult,
    GuardIsolationData,
    SimulatorOptions,
    SimulatorState,
    SimulationResults,
    ResultsOptions,
)

from .ode_solver import ODESolver
from .solution import (
    make_empty_solution,
    update_sim_solution,
)
from ..framework import is_event_data, ZeroCrossingEvent

if TYPE_CHECKING:
    from .types import ODESolverOptions, ODESolution
    from .ode_solver import ODESolverBase
    from ..framework import ContextBase, SystemBase
    from ..framework.cache import CacheSource
    from ..framework.event import PeriodicEventData, CompositeEventCollection


def _raise_end_time_not_reached(tf, ctx_time):
    if (tf - ctx_time) / tf > 1e-3:
        raise RuntimeError(
            f"Simulator failed to reach specified end time. End time={tf}. Reached time={ctx_time}. Try increasing Maximum Major Steps."
        )


@jax.jit
def error_end_time_not_reached(tf, ctx_time):
    jax.debug.callback(_raise_end_time_not_reached, tf, ctx_time)


def _simulate(
    system: SystemBase,
    context: ContextBase,
    tspan: tuple[float, float],
    options: SimulatorOptions = None,
    results_options: ResultsOptions = None,
    ode_options: ODESolverOptions = None,
    # TODO: deprecate (now a part of SimulatorOptions)
    recorded_signals: dict[str, CacheSource] = None,
) -> SimulationResults:
    # This is the JITable version of `simulate`, which does not include any
    # postprocessing.  It should typically not be called directly.

    if recorded_signals is not None:
        options.recorded_signals = recorded_signals
    options.save_time_series = options.recorded_signals is not None

    if results_options is None:
        results_options = ResultsOptions()

    t0, tf = tspan
    initial_context = context.with_time(t0)

    ode_solver = ODESolver(system, options=ode_options)
    sim = Simulator(system, ode_solver=ode_solver, simulator_options=options)
    logger.debug("Simulator is configured with max_major_steps=%s", sim.max_major_steps)

    sim_state = sim.advance_to(tf, initial_context)

    error_end_time_not_reached(tf, sim_state.context.time)

    final_context = sim_state.context if options.return_context else None
    solution = sim_state.solution

    return SimulationResults(final_context, data=solution)


def _check_tracing_enabled(system: SystemBase, options: SimulatorOptions) -> bool:
    enable_tracing = True if options is None else options.enable_tracing
    return (
        enable_tracing
        and system.enable_trace_cache_sources
        and system.enable_trace_unrestricted_updates
        and system.enable_trace_discrete_updates
    )


def simulate(
    system: SystemBase,
    context: ContextBase,
    tspan: tuple[float, float],
    options: SimulatorOptions = None,
    results_options: ResultsOptions = None,
    ode_options: ODESolverOptions = None,
    # TODO: deprecate (now a part of SimulatorOptions)
    recorded_signals: dict[str, CacheSource] = None,
    postprocess: bool = True,
) -> SimulationResults:
    """Simulate the hybrid dynamical system defined by `system`.

    The parameters and initial state are defined by `context`.  The simulation time
    runs from `tspan[0]` to `tspan[1]`.

    The default behavior is to advance and return the final context.  If
    `recorded_signals` is provided as a dict of (name, signal_source),
    the simulation will also return a solution dict mapping from name to
    the time series of the signal, along with an entry for `"time"`.
    Typically the signal sources will be output ports, but they can be
    any `CacheSource` object.

    Configuration options can be provided via the `options`, `results_options`,
    and `ode_options` arguments.  See each dataclass definition for possible options.
    If `recorded_signals` is provided as a kwarg, it will override any entry in
    `options.recorded_signals`. This should be deprecated in the future in favor of
    only passing `options`.
    """
    if options is None:
        options = SimulatorOptions()

    if options.max_major_steps is None or options.max_major_steps <= 0:
        options.max_major_steps = SimulatorOptions.calculate_max_major_steps(
            system, tspan
        )

    def _wrapped_simulate() -> SimulationResults:
        results = _simulate(
            system,
            context,
            tspan,
            ode_options=ode_options,
            options=options,
            results_options=results_options,
            recorded_signals=recorded_signals,
        )
        return results

    # Check if the simulation can be traced (i.e. if it is JIT-able)
    if _check_tracing_enabled(system, options):
        _wrapped_simulate = jax.jit(_wrapped_simulate)
        _wrapped_simulate = Profiler.jaxjit_profiledfunc(
            _wrapped_simulate, "_wrapped_simulate"
        )

    results = _wrapped_simulate()

    # This will trim the solution data if there are unused buffer entries.
    # Since it returns variable-length arrays, it cannot be traced or JIT-compiled.
    # If it is necessary to traced the entire call to `simulate` (e.g. with vmap), then
    # postprocessing should be deferred until after the simulation is complete.
    if postprocess and results.data is not None:
        results = results.postprocess()

    return results


class Simulator:
    def __init__(
        self,
        system: SystemBase,
        ode_solver: ODESolverBase = None,
        simulator_options: SimulatorOptions = None,
    ):
        self.system = system

        if simulator_options is None:
            simulator_options = SimulatorOptions()

        if ode_solver is None:
            ode_solver = ODESolver(system)

        self.iso_scale_factor = simulator_options.guard_isolation_scale_factor
        self.max_steps = ode_solver.max_steps
        self.max_major_steps = simulator_options.max_major_steps
        self.max_major_step_length = simulator_options.max_major_step_length
        self.save_time_series = simulator_options.save_time_series
        self.recorded_outputs = simulator_options.recorded_signals

        # Determine whether JAX tracing can be used (jit, grad, vmap, etc)
        self.enable_tracing = _check_tracing_enabled(system, simulator_options)

        logger.debug("Simulator created with enable_tracing=%s", self.enable_tracing)

        self._initialization_done = False
        self.ode_solve = ode_solver

        # Modify the default autodiff rule slightly to correctly capture variations
        # in end time of the simulation interval.
        self.advance_to = self._override_advance_to_vjp()

        # Automatically determine a tolerance for localizing zero-crossing events
        eps = ode_solver.rtol * self.iso_scale_factor * self.system.characteristic_time
        self.trigger_search = self._override_trigger_search_jvp(eps)

    def while_loop(self, cond_fun, body_fun, val):
        if not self.enable_tracing:
            while cond_fun(val):
                val = body_fun(val)
            return val
        else:
            return lax.while_loop(
                cond_fun,
                body_fun,
                val,
            )

    def cond(self, pred, true_fun, false_fun, args=()):
        if not self.enable_tracing:
            return true_fun(*args) if pred else false_fun(*args)
        else:
            return lax.cond(pred, true_fun, false_fun, *args)

    def for_loop(self, lower: SupportsIndex, upper: SupportsIndex, body_fun, args):
        if not self.enable_tracing:
            for i in range(lower, upper):
                args = body_fun(i, args)
            return args
        else:
            return lax.fori_loop(lower, upper, body_fun, args)

    def initialize(self, context: ContextBase) -> SimulatorState:
        logger.debug("Initializing simulator")
        # context.state.pprint(logger.debug)

        # Recursively delete as much of the cache as possible from all contexts
        context = context.clear_cache()

        current_time = context.time

        # Ensure that next_update_time() can return the current time by perturbing
        # current time as slightly toward negative infinity as possible
        # TODO: Is this necessary?  https://github.com/RobotLocomotion/drake/issues/13296
        t_eps = jnp.finfo(jnp.result_type(context.time)).eps
        context = context.with_time(current_time - t_eps)

        time_of_next_timed_event, timed_events = next_update_time(
            self.system.periodic_events, context.time
        )

        # timed_events is now marked with the active events at the next update time
        logger.debug(f"Time of next timed event: {time_of_next_timed_event}")
        timed_events.pprint(logger.debug)

        trigger_type = jnp.where(
            time_of_next_timed_event == current_time,
            TimeOrWitnessTriggered.kTimeTriggered,
            TimeOrWitnessTriggered.kNothingTriggered,
        )

        # Reset time
        context = context.with_time(current_time)

        return SimulatorState(
            context=context,
            trigger_type=trigger_type,
            timed_events=timed_events,
            solution=None,
        )

    def _override_advance_to_vjp(self):
        # If JAX tracing is enabled for autodiff, wrap the advance function with a
        # custom autodiff rule to correctly capture variation with respect to end
        # time. If somehow autodiff works with tracing disabled, the derivatives will
        # not account for possible variations in end time (for instance in finding
        # limit cycles or when there are terminal conditions on the simulation).

        if not self.enable_tracing:
            return self._advance_to

        # This is the main function call whose autodiff rule will be overridden.
        def _wrapped_advance_to(
            sim: Simulator, boundary_time: float, context: ContextBase
        ) -> SimulatorState:
            return sim._advance_to(boundary_time, context)

        # The "forwards pass" to advance the simulation.  Also stores the nominal
        # VJP calculation and the continuous time derivative value, both of which
        # will be needed in the backwards pass.
        def _wrapped_advance_to_fwd(
            sim: Simulator, boundary_time: float, context: ContextBase
        ) -> tuple[SimulatorState, tuple]:
            primals, vjp_fun = jax.vjp(sim._advance_to, boundary_time, context)

            # Also need to keep the final continuous time derivative value for
            # computing the adjoint time variable
            xdot = sim.system.eval_time_derivatives(primals.context)

            # "Residual" information needed in the backwards pass
            res = (vjp_fun, xdot)

            return primals, res

        def _wrapped_advance_to_adj(
            _sim: Simulator, res: tuple, adjoints: SimulatorState
        ) -> tuple[float, ContextBase]:
            # Unpack the residuals from the forward pass
            vjp_fun, xdot = res

            # Compute whatever the standard adjoint variables are using the
            # automatically derived VJP function.  The first return variable will
            # be the automatically computed tf_adj value, which we will ignore in
            # favor of the manually derived value computed next.
            _, context_adj = vjp_fun(adjoints)

            # The derivative with respect to end time is just the dot product of
            # the adjoint "seed" continuous state with the final time derivatives.
            # We can overwrite whatever the calculated adjoint time was with this.
            vc = adjoints.context.continuous_state
            vT_xdot = jax.tree_map(lambda xdot, vc: jnp.dot(xdot, vc), xdot, vc)
            tf_adj = sum(jax.tree_util.tree_leaves(vT_xdot))

            # Return adjoints to match the inputs to _wrapped_advance_to, except for
            # the first argument (Simulator), which will be marked nondifferentiable.
            return (tf_adj, context_adj)

        advance_to = jax.custom_vjp(_wrapped_advance_to, nondiff_argnums=(0,))
        advance_to.defvjp(_wrapped_advance_to_fwd, _wrapped_advance_to_adj)

        # Copy the docstring and type hints to the overridden function
        advance_to.__doc__ = self._advance_to.__doc__
        advance_to.__annotations__ = self._advance_to.__annotations__

        return partial(advance_to, self)

    def _override_trigger_search_jvp(self, eps):
        # If JAX tracing is enabled for autodiff, wrap the zero-crossing localization
        # function with a custom autodiff rule to differentiate through the result of
        # the while loop

        def _wrapped_trigger_search(
            sim: Simulator,
            tf: float,
            guards: CompositeEventCollection,
            context: ContextBase,
        ) -> tuple[GuardIsolationData, ODESolution]:
            logger.debug("IN PRIMAL")
            return sim._trigger_search(tf, guards, context, eps=eps)

        if not self.enable_tracing:
            return partial(_wrapped_trigger_search, self, eps=eps)

        def _wrapped_trigger_search_fwd(
            simulator: Simulator,
            tf: float,
            guards: CompositeEventCollection,
            context: ContextBase,
        ) -> tuple[tuple[GuardIsolationData, ODESolution], tuple]:
            # JAX can't do reverse-mode AD through a while loop.  However, since the while
            # loop is repeating the solve from the same starting point, we could effectively
            # overwrite the jvp_fun each time through the loop and just keep the last one
            # from _within_ the base `_trigger_search` function
            #
            # So far this doesn't work because the vjp_fun is registered as a different type
            # in the PyTree structure

            # A simple, but slightly inefficient alternative is to just do one final vjp call
            # at the end of the loop.  Since we know that the condition function in `while`
            # will evaluate to `True`` with these values, we can just directly evaluate the
            # final vjp function.
            logger.debug("IN FORWARD PASS")

            t0 = context.time
            x0 = context.state

            search_data, ode_sol = _wrapped_trigger_search(
                simulator, tf, guards, context
            )

            # Modify `tf` so that `_body_fun` will compute the correct value
            # based on the bisection search rule `tf = 2.0*tc - t0`
            search_data = search_data._replace(tf=2.0 * search_data.tf - t0)

            # Call the function one more time in order to produce the vjp function
            # to pass as a residual for the backwards pass.
            # To do this, match the arguments to _bisection_body_fun:
            _body_fun = partial(_bisection_body_fun, simulator.ode_solve)
            guards = search_data.guards
            args = (x0, t0, search_data.guards, (search_data, ode_sol))
            primals_out, vjp_fun, ode_sol = jax.vjp(_body_fun, *args, has_aux=True)

            # Return the primal values as well as `ode_sol` to match the nominal outputs
            logger.debug("DONE WITH FORWARD PASS")
            residuals = (vjp_fun,)
            return (primals_out, ode_sol), residuals

        def _wrapped_trigger_search_adj(
            _simulator: Simulator,
            residuals: tuple,
            adjoints: tuple[GuardIsolationData, ODESolution],
        ) -> tuple[float, CompositeEventCollection, ContextBase]:
            (vjp_fun,) = residuals

            data_adj, _ = adjoints  # Adjoint state includes aux data for `ode_sol`

            # vjp_fun outputs have the same structure as inputs to _bisection_body_fun
            x0_adj, t0_adj, guards_adj, carry_adj = vjp_fun(data_adj)
            adjoints_out: GuardIsolationData = carry_adj[0]

            # Before the while loop/body function, the inputs are unmodified.  So after
            # passing through the vjp function, we just have to account for the initial
            # modification to tf and then recover the inputs.
            tf_adj = 2.0 * adjoints_out.tf  # Derivative of the bisection initialization

            guards_adj = adjoints_out.guards
            context_adj = adjoints_out.context.with_state(x0_adj)
            context_adj = context_adj.with_time(t0_adj)
            return tf_adj, guards_adj, context_adj

        # Associate the custom vjp rules with the nominal function.
        trigger_search = jax.custom_vjp(_wrapped_trigger_search, nondiff_argnums=(0,))
        trigger_search.defvjp(_wrapped_trigger_search_fwd, _wrapped_trigger_search_adj)

        # Copy the docstring and type hints to the overridden function
        trigger_search.__doc__ = self._trigger_search.__doc__
        trigger_search.__annotations__ = self._trigger_search.__annotations__

        return partial(trigger_search, self)

    def _guarded_integrate(self, context: ContextBase, tf):
        # Save the time and current state
        logger.debug(
            "Integrating from t=%s with xc=%s, xd=%s",
            context.time,
            context.continuous_state,
            context.discrete_state,
        )

        guards = self.system.determine_active_guards(context)

        # Record guard function values at t0
        guards = guard_interval_start(guards, context)

        # Integrate the ODEs, stopping short of any state transition events.  If there are no
        #   state transitions, this will integrate to tf.
        # search_data, ode_sol = self.trigger_search(tf, guards, context)
        result = self.trigger_search(tf, guards, context)
        logger.debug(result)
        search_data, ode_sol = result
        triggered, tf, guards, context = search_data

        return triggered, context, ode_sol, guards

    def _unguarded_integrate(self, context: ContextBase, tf):
        # NOTE: This is really just a special case of guarded_integrate where
        # the guard collection is empty.  It's separated out for development
        # convenience, but should be merged back in later once all the simulation
        # logic and control flow is stable.

        # Integrate the ODE to the final time of the major_step
        t_span = (context.time, tf)
        ode_sol, context = self.ode_solve(context, x0=context.state, t_span=t_span)

        guards = self.system.determine_active_guards(context)
        return False, context, ode_sol, guards

    # @partial(jax.custom_vjp, nondiff_argnums=(0, 4))
    def _trigger_search(self, tf, guards, context, eps=1e-8):
        # From Drake docs:
        # Determines whether any witnesses trigger over the interval [t0, tw],
        # where tw - t0 < ε and ε is the "witness isolation length". If one or more
        # witnesses does trigger over this interval, the time (and corresponding state)
        # will be advanced to tw and those witnesses will be stored in
        # `triggered_witnesses` on return. On the other hand (i.e., if no witnesses)
        # trigger over [t0, t0 + ε], time (and corresponding state) will be advanced
        # to some tc in the open interval (t0, tf) such that no witnesses trigger
        # over [t0, tc]; in other words, we deem it "safe" to integrate to tc.
        # @param[in,out] triggered_witnesses on entry, the set of witness functions
        #             that triggered over [t0, tf]; on exit, the set of witness
        #             functions that triggered over [t0, tw], where tw is some time
        #             such that tw - t0 < ε. If no functions trigger over
        #             [t0, t0 + ε], `triggered_witnesses` will be empty on exit.
        # @pre The time and state are at tf and x(tf), respectively, and at least
        #     one witness function has triggered over [t0, tf].
        # @post If `triggered_witnesses` is empty, the time and state will be
        #     set to some tc and x(tc), respectively, such that no witnesses trigger
        #     over [t0, tc]. Otherwise, the time and state will be set to tw and
        #     x(tw), respectively.
        # @note The underlying assumption is that a witness function triggers over a
        #     interval [a, d] for d ≤ the maximum integrator step size if that
        #     witness also triggers over interval [a, b] for some b < d. Per
        #     WitnessFunction documentation, we assume that a witness function
        #     crosses zero at most once over an interval of size [t0, tf]).

        t0 = context.time
        x0 = context.state

        # From Drake:
        # / TODO(edrumwri): Speed this process using interpolation between states,
        # // more powerful root finding methods, and/or introducing the concept of
        # // a dead band.

        # Bisection-type search for trigger over [t0, tc] starting from t_mid=(t0+tf)/2.
        # If nothing triggers for a particular tc, this indicates that it is safe to
        # integrate from (t0, tc) without guards triggering.
        # Then attempt to integrate from tc to tf, in which we know there is a guard
        # trigger.
        def _cond_fun(carry):
            data, _ = carry
            triggered, tc = data.triggered, data.tf
            return triggered & ((tc - t0) > eps)

        _body_fun = partial(_bisection_body_fun, self.ode_solve, x0, t0, guards)

        # Initial loop values that will get filled in by the first forward call
        # Have to split up the data into a tuple and `ode_sol` because we don't
        # want to treat the solution data as directly differentiable.
        carry = (GuardIsolationData(None, 2.0 * tf - t0, None, context), None)

        # Initial integration step and fill in empty initial carry values
        carry = _body_fun(carry)

        # Loop until an endpoint is found that is before or at a triggered guard
        return self.while_loop(_cond_fun, _body_fun, carry)

    def _advance_to(self, boundary_time: float, context: ContextBase) -> SimulatorState:
        """Core control flow logic for running a simulation.

        This is marked private because it is wrapped with a custom autodiff rule to
        get the correct derivatives with respect to end time. Normally the wrapped
        attribute `advance_to` is what should be called by users.
        """
        # TODO: Use checkify to catch errors here

        system = self.system
        sim_state = self.initialize(context)
        trigger_type = sim_state.trigger_type
        context = sim_state.context
        timed_events = sim_state.timed_events

        if self.save_time_series:
            solution = make_empty_solution(
                context,
                recorded_signals=self.recorded_outputs,
                max_steps=self.max_steps,
                max_major_steps=self.max_major_steps,
            )
        else:
            solution = None

        # Only activate timed events if the major step ended on a time trigger
        timed_events = activate_timed_events(timed_events, trigger_type)

        def _false_fun(sim_state: SimulatorState) -> SimulatorState:
            return sim_state

        # TODO: Give this a better name
        def _true_fun(sim_state: SimulatorState) -> SimulatorState:
            trigger_type = sim_state.trigger_type
            context = sim_state.context
            timed_events = sim_state.timed_events
            solution = sim_state.solution

            if not self.enable_tracing:
                logger.debug("Starting a simulation step at t=%s", context.time)
                logger.debug("   merged_events: %s", timed_events)

            # NOTE: This logic to test for triggered guards before/after updates
            # doesn't exist in Drake.  It's necessary for instance for exiting a
            # Zeno state from an Integrator if a change is triggered by a timed
            # update event.  This `transition_events` object is also separate from
            # the one returned from `_(un)guarded_integrate`.
            #
            # Get the set of guards before updates
            transition_events = system.determine_active_guards(context)
            if transition_events.has_events:
                # Record guard values at t0
                transition_events = guard_interval_start(transition_events, context)

            context = system.handle_unrestricted_update(timed_events, context)
            context = system.handle_discrete_update(timed_events, context)

            if transition_events.has_events:
                # Check if guards have triggered as a result of these updates
                transition_events = guard_interval_end(transition_events, context)
                transition_events = determine_triggered_guards(
                    transition_events, context
                )

                # Handle any triggered guards
                context = system.handle_zero_crossings(transition_events, context)

            # How far can we go before we have to handle timed events? This can return
            # infinity, meaning we don't see any timed events coming. When an earlier
            # event trigger time is returned, at least one Event object must be
            # returned. Note that if the returned time is the current time, we handle
            # the Events and then restart at the same time, possibly discovering more
            # events.
            time_of_next_timed_event, timed_events = next_update_time(
                system.periodic_events, context.time
            )
            # assert time_of_next_timed_event >= step_start_time
            if not self.enable_tracing:
                logger.debug("Next timed event at t=%s", time_of_next_timed_event)
                timed_events.pprint(logger.debug)

            # Determine whether the events requested includes an Update, a Publish, or both
            update_time = jnp.array(jnp.inf, dtype=time_of_next_timed_event.dtype)
            publish_time = jnp.array(jnp.inf, dtype=time_of_next_timed_event.dtype)

            if (timed_events.num_discrete_update_events > 0) or (
                timed_events.num_unrestricted_update_events > 0
            ):
                update_time = time_of_next_timed_event
            if timed_events.num_publish_events > 0:
                publish_time = time_of_next_timed_event

            # Limit the major step end time to the simulation end time, or major step limit.
            # This is the mechanism used to advance time for systems that have
            # no states and no periodic events.
            end_time = jnp.min(
                jnp.array([boundary_time, context.time + self.max_major_step_length])
            )
            if not self.enable_tracing:
                logger.debug(
                    "Integrating to one of: publish_time=%s, update_time=%s, end_time=%s",
                    publish_time,
                    update_time,
                    end_time,
                )

            result, tf = _termination_reason(publish_time, update_time, end_time)
            logger.debug("Expecting to integrate to t=%s for reason %s", tf, result)

            # determine if model is either feedforward or purely discrete
            # Note: this only matters for the results saving. when there are
            # continuous states or unrestricted events, we expect that calling
            # _(un)guarded_integrate() will advance time, and have non-zero time
            # stamped solution samples temporarily stored in ode_sol. When this
            # happens, these time stamped solution samples are then extended with
            # discrete state values "as the system would see them", for this major
            # step.
            # When there are no continuous states nor unrestricted events, then
            # _(un)guarded_integrate() will only return a time stamped solution sample
            # at the end time of the major step. extending this with the discrete
            # state values "as the system would see them", means that the results
            # returned to the user seem to be delayed in time by one discrete step.
            # to get around this, we simply advance time only after the we extend
            # the solution object with the value of the discrete state/outputs.
            not_ffwd_or_discrete = context.num_continuous_states > 0

            if transition_events.has_events:
                _integrate = self._guarded_integrate
            else:
                _integrate = self._unguarded_integrate

            if not_ffwd_or_discrete:
                triggered, context, ode_sol, transition_events = _integrate(context, tf)
            else:
                # no need to modify context nor transition_events in the
                # case of purely discrete system
                triggered = False
                ode_sol = None

            trigger_type = _determine_trigger_type(
                triggered, tf, update_time, publish_time
            )
            logger.debug("Returning trigger: %s", trigger_type)

            if self.save_time_series:
                solution = update_sim_solution(
                    context, solution, self.recorded_outputs, ode_sol
                )

            if not not_ffwd_or_discrete:
                # When there are no continuous states, we dont use ode_sol
                # which contains all solver minor steps, to create the solution
                # samples.
                # Discrete systems] when there are discrete periodic events, we use those to determine
                # each major step end time.
                # Feedforward system] when there are just feed forward block, no states, no events, we
                # use max_major_step_length to determine each major step end time.
                # in the end, we use 'tf' here because that is the culmination of
                # the discrete events and max_major_step_length controls.

                # additionally, somewhat unfortunate, but in the case of Feedforward
                # systems, we advance time like ctx.time+dt. this floating op results in
                # drift, so although we expect times=range(0,end_time+dt,dt), it can happen
                # that:
                #   for n in range(floor(end_time/dt),
                #       ctx.time = ctx.time+dt
                #   ctx.time < end_time
                # the outcome is that we get a results sample at end_time-eps, and another
                # at end_time.
                # to avoid this, we do the check below.
                # FIXME: 1e-9 is a HACK. we should use a tolerance that is dynamic w.r.t.
                # the time scales of the simulation. since this is only for Feedforward
                # systems, it is ok for now.
                tf_ = self.cond(
                    (boundary_time - tf) < 1e-9, lambda: boundary_time, lambda: tf
                )
                context = context.with_time(tf_)

            # TODO: Constraint projection (not implemented in Drake either)

            # Only activate timed events if the major step ended on a time trigger
            timed_events = activate_timed_events(timed_events, trigger_type)

            # Handle any triggered guards
            context = system.handle_zero_crossings(transition_events, context)

            return SimulatorState(
                trigger_type=trigger_type,
                context=context,
                solution=solution,
                timed_events=timed_events,
            )

        def _body_fun(i, sim_state):
            pred = sim_state.context.time < boundary_time
            return self.cond(pred, _true_fun, _false_fun, (sim_state,))

        sim_state = SimulatorState(
            trigger_type=trigger_type,
            context=context,
            solution=solution,
            timed_events=timed_events,
        )

        logger.debug(
            "Running simulation from t=%s to t=%s", context.time, boundary_time
        )
        try:
            sim_state = self.for_loop(0, self.max_major_steps, _body_fun, sim_state)
            logger.debug("Simulation complete at t=%s", sim_state.context.time)
        except KeyboardInterrupt:
            # TODO: flag simulation as interrupted somewhere in sim_state
            logger.info("Simulation interrupted at t=%s", sim_state.context.time)

        # update discrete state to x+ at the simulation end_time
        # 1] do discrete update
        # 2] do update solution
        if self.save_time_series:
            logger.debug("Finalizing solution...")
            context = system.handle_discrete_update(
                sim_state.timed_events, sim_state.context
            )
            solution = update_sim_solution(
                context, sim_state.solution, self.recorded_outputs, None
            )
            sim_state = sim_state._replace(context=context)
            sim_state = sim_state._replace(solution=solution)
            logger.debug("Done finalizing solution")

        return sim_state


@jax.jit
def _termination_reason(publish_time, update_time, boundary_time):
    # See documentation for IntegratorBase::IntegrateNoFurtherThanTime in Drake for
    # an explanation of the ordering and logic here.

    candidate_result = jnp.where(
        publish_time < update_time,
        StepResult.kReachedPublishTime,
        StepResult.kReachedUpdateTime,
    )
    target_time = jnp.where(
        publish_time < update_time,
        publish_time,
        update_time,
    )

    candidate_result = jnp.where(
        boundary_time < target_time,
        StepResult.kReachedBoundaryTime,
        candidate_result,
    )
    target_time = jnp.where(
        boundary_time < target_time,
        boundary_time,
        target_time,
    )

    return candidate_result, target_time


@jax.jit
def _determine_trigger_type(guard_triggered, tf, update_time, publish_time):
    # Determine trigger type assuming that a guard triggers over the sub-interval.
    # If the integration terminated due to a triggered event, determine whether there are any
    #    other events that should be triggered at the same time.
    _guard_trigger_type = jnp.where(
        ((tf == update_time) | (tf == publish_time)),
        TimeOrWitnessTriggered.kBothTriggered,
        TimeOrWitnessTriggered.kWitnessTriggered,
    )

    # No guard triggered; handle integration as usual.
    _no_guard_trigger_type = jnp.where(
        ((tf == update_time) | (tf == publish_time)),
        TimeOrWitnessTriggered.kTimeTriggered,
        TimeOrWitnessTriggered.kNothingTriggered,
    )

    return jnp.where(
        guard_triggered,
        _guard_trigger_type,
        _no_guard_trigger_type,
    )


def _next_sequence_time(current_time: float, period: float, offset: float) -> float:
    """Compute the index in the sequence of samples for the next time to sample,
    which should be greater than the present time.  This allows the return time
    to be less than offset (should be wrapped by the conditional in _next_sample_time)
    """

    # If the period is infinite (i.e. a one-time event), then right after the update next_k = 0
    # and 0.0 * inf = nan, so we need to handle this case separately.

    offwith_time = current_time - offset
    next_k = jnp.ceil(offwith_time / period)

    next_t = jnp.where(
        jnp.isfinite(period),
        offset + next_k * period,
        period,
    )

    return jnp.where(next_t <= current_time, offset + (next_k + 1) * period, next_t)


def _next_sample_time(current_time: float, event_data: PeriodicEventData) -> float:
    """Return max(offset, next_sequence_time(current_time))"""
    period, offset = event_data.period, event_data.offset
    return jnp.where(
        current_time < offset,
        offset,
        _next_sequence_time(current_time, period, offset),
    )


def next_update_time(periodic_events: CompositeEventCollection, current_time) -> float:
    """Compute next update time over all events in the periodic_events collection.

    System::DoCalcNextUpdateTime in v1/Drake

    This returns a tuple of the minimum next sample time along with a pytree with
    the same structure as `periodic_events` indicating which events are active at
    the next sample time.
    """
    periodic_events = periodic_events.mark_all_inactive()

    # 0. If no events, return an infinite time and empty event collection
    if not periodic_events.has_events:
        return jnp.array(jnp.inf), periodic_events

    # 1. Compute the next sample time for each event
    def _replace_sample_time(event_data):
        return dataclasses.replace(
            event_data,
            next_sample_time=_next_sample_time(current_time, event_data),
        )

    timed_events = jax.tree_map(
        _replace_sample_time,
        periodic_events,
        is_leaf=is_event_data,
    )

    def _get_next_sample_time(event_data: PeriodicEventData) -> float:
        return event_data.next_sample_time

    # 2. Find the minimum next sample time across all events
    min_time = jax.tree_util.tree_reduce(
        jnp.minimum,
        jax.tree_map(
            _get_next_sample_time,
            timed_events,
            is_leaf=is_event_data,
        ),
    )

    # 3. Find the events corresponding to the minimum time by updating the event data `active` field
    def _replace_active(event_data: PeriodicEventData):
        return dataclasses.replace(
            event_data,
            active=(event_data.next_sample_time == min_time),
        )

    active_events = jax.tree_map(
        _replace_active,
        timed_events,
        is_leaf=is_event_data,
    )
    return min_time, active_events


def activate_timed_events(
    timed_events: CompositeEventCollection, trigger_type: TimeOrWitnessTriggered
) -> CompositeEventCollection:
    # Only activate timed events if the major step ended on a time trigger
    deactivate = trigger_type != TimeOrWitnessTriggered.kTimeTriggered

    def activation_fn(event_data: PeriodicEventData):
        return event_data.active & ~deactivate

    return timed_events.activate(activation_fn)


def _record_guard_values(
    events: CompositeEventCollection, context: ContextBase, key: str
) -> CompositeEventCollection:
    # Set the `w0`/`w1` field of event_data by evaluating the guard functions
    def _update(event: ZeroCrossingEvent):
        event.event_data = dataclasses.replace(
            event.event_data,
            **{key: event.eval_guard(context)},
        )
        return event

    def _is_transition(x):
        return isinstance(x, ZeroCrossingEvent)

    return jax.tree_map(_update, events, is_leaf=_is_transition)


guard_interval_start = partial(_record_guard_values, key="w0")
guard_interval_end = partial(_record_guard_values, key="w1")


def determine_triggered_guards(
    events: CompositeEventCollection, context: ContextBase
) -> CompositeEventCollection:
    events = guard_interval_end(events, context)

    def _update(event: ZeroCrossingEvent):
        event.event_data = dataclasses.replace(
            event.event_data,
            active=event.should_trigger(),
        )
        return event

    def _is_transition(x):
        return isinstance(x, ZeroCrossingEvent)

    return jax.tree_map(_update, events, is_leaf=_is_transition)


def _bisection_body_fun(ode_solve, x0, t0, guards, carry):
    data, _ = carry  # Ignore the current ode_sol value - will overwrite here.
    tf, context = data.tf, data.context
    tc = (t0 + tf) / 2  # Rule for bisection search in time.
    ode_sol, context = ode_solve(context, x0=x0, t_span=(t0, tc))
    guards = determine_triggered_guards(guards, context)
    triggered = guards.has_active
    return GuardIsolationData(triggered, tc, guards, context), ode_sol
