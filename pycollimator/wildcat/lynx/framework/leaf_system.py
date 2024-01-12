"""Self-contained systems with no subsystems.

A LeafSystem is a minimal component of a system model in wildcat, containing no
subsystems.  If it has inputs or outputs it can be connected to other LeafSystems
as "blocks" to form Diagrams.  If not, it is a self-contained dynamical system.

The LeafSystem class defines the interface for these blocks, specifying various
options for configuring the block.  For example, a LeafSystem can declare that
it has a continuous state and provide an ODE function governing the time evolution
of that state.  It can also declare discrete states, parameters, and update events.
The built-in blocks in wildcat.library are all subclasses of LeafSystem, as are
any custom blocks defined by the user.

After declaring states, parameters, ODE, updates, etc., the LeafSystem comprises a set
of pure functions that can be evaluated given a Context, which contains the actual
numeric values for the states, parameters, etc.
"""
from __future__ import annotations
from functools import partial
from typing import List, Set, Tuple, Callable, TYPE_CHECKING, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..logging import logger
from ..math_backend import (
    array,
    zeros,
    zeros_like,
    cond,
)

from .state import LeafState

from .system_base import SystemBase, InstanceParameter
from .context_factory import LeafContextFactory
from .dependency_graph import (
    mark_cache,
    LeafDependencyGraphFactory,
    DependencyTicket,
    next_dependency_ticket,
)

from .event import (
    LeafCompositeEventCollection,
    PublishEvent,
    PeriodicEventData,
    DiscreteUpdateEvent,
    UnrestrictedUpdateEvent,
    ZeroCrossingEvent,
    ZeroCrossingEventData,
)

if TYPE_CHECKING:
    from ..math_backend.typing import (
        Array,
        Scalar,
        ShapeLike,
        DTypeLike,
    )
    from .context import ContextBase, LeafContext
    from .cache import CacheSource

__all__ = ["LeafSystem"]


# Helper functons used in feedthrough determination
_mark_up_to_date = partial(mark_cache, is_out_of_date=False)
_mark_out_of_date = partial(mark_cache, is_out_of_date=True)


class LeafSystem(SystemBase):
    """Basic building block for dynamical systems.

    A LeafSystem is a minimal component of a system model in wildcat, containing no
    subsystems.  Inputs, outputs, state, parameters, updates, etc. can be added to the
    block using the various `declare_*` methods.  The built-in blocks in
    wildcat.library are all subclasses of LeafSystem, as are any custom blocks defined
    by the user."""

    # SystemBase is a dataclass, so we need to call __post_init__ explicitly
    def __post_init__(self):
        super().__post_init__()
        logger.debug(f"Initializing {self.name} [{self.system_id}]")

        # If not None, this defines the shape and data type of the continuous state
        # component.  This value will be used to initialize the context, so it will
        # also serve as the initial value unless explicitly overridden. It will
        # typically be an array, but it can be any PyTree-structured object (list,
        # dict, namedtuple, etc.), provided that the ODE function returns a PyTree
        # of the same structure.
        self._default_continuous_state: Array = None

        # The callback for computing the RHS of the ODE is stored in a CacheSource.
        # Must return a value with the same shape and dtype as the continuous state.
        self._ode_cache: CacheSource = None

        # If not empty, this defines the shape and data type of the discrete states.
        # This value will be used to initialize the context, so it will also serve
        # as the initial value unless explicitly overridden.  In contrast to the single
        # continuous state, a LeafSystem in general may have a group of discrete
        # states, since they may be updated in different ways.  For example, one
        # discrete state component may be updated periodically, while another is only
        # updated as a result of a zero-crossing event.
        self._default_discrete_state: List[Array] = []

        # If not None, the system has a "mode" or "stage" component of the state.
        # In a "state machine" paradigm, this represents the current state of the
        # system (although "state" is obviously used for other things in this case).
        # The mode is an integer value, and the system can declare transitions between
        # modes using the `declare_zero_crossing` method, which in addition to the
        # guard function and reset map also takes optional `start_mode` and `end_mode`
        # arguments.
        self._default_mode: int = None

        # DependencyTickets for the discrete state components.  These will be
        # automatically generated when the discrete states are declared. Users
        # should not have to interact with these directly.
        self._discrete_state_tickets: List[DependencyTicket] = []

        # If not empty, this defines the shape and data type of the numeric parameters
        # in the LeafSystem. This value will be used to initialize the context, so it
        # will also serve as the initial value unless explicitly overridden. In the
        # simplest cases, parameters could be stored as attributes of the LeafSystem,
        # but declaring them has the advantage of moving the values to the context,
        # allowing them to be traced by JAX rather than stored as static data. This
        # means they can be differentiated, vmapped, or otherwise modified without
        # re-compiling the simulation.
        self._default_parameters: dict[str, Array] = {}

        # DependencyTickets for the parameters.  These will be automatically
        # generated when the parameters are declared. Users should not have to
        # interact with these directly.
        self._parameter_tickets: dict[str, DependencyTicket] = {}

        # Transition map from (start_mode -> [*end_modes]) indicating which
        # transition events are active in each mode.  This is not used by
        # any logic in the system, but can be useful for debugging.
        self.transition_map: dict[int, List[Tuple[int, ZeroCrossingEvent]]] = {}

        # Set of events that updates at a fixed rate.  Each event has its own period
        # and offset, so "fires" independently of the other events. These can be
        # created using the `declare_periodic_discrete_update` and
        # `declare_periodic_unrestricted_update` methods.
        self._periodic_events = LeafCompositeEventCollection()

        # Set of events that update when a zero-crossing occurs.  Each event has its
        # own guard function and, optionally, reset map, start mode, and end mode.
        # These can be created using the `declare_zero_crossing` method.
        self._zero_crossing_events = LeafCompositeEventCollection()

    #
    # Serialization
    #
    @property
    def default_parameters(self) -> dict[str, Array]:
        return self._default_parameters

    #
    # Event handling
    #
    def wrap_update_callback(
        self, callback: Callable, collect_inputs: bool = True
    ) -> Callable:
        """Wrap an update function to unpack local variables and block inputs.

        The callback should have the signature
        `callback(time, state, *inputs, **params) -> result`
        and will be wrapped to have the signature `callback(context) -> result`,
        as expected by the event handling logic.

        This is used internally for declaration methods like
        `declare_periodic_discrete_update` so that users can write more intuitive
        block-level update functions without worrying about the "context", and have
        them automatically wrapped to have the right interface.  It can also be
        called directly by users to wrap their own update functions, for example to
        create a callback function for `declare_output_port`.

        TODO: LeafSystem.declare_output_port should actually call this itself, so that
        users never need to write "context callbacks", which is confusing. See:
        https://collimator.atlassian.net/browse/WC-119

        The context and state are strictly immutable, so the callback should not
        attempt to change any values in the context or state.  Even in cases where
        it is impossible to _enforce_ this (e.g. a state component is a list, which
        is always mutable in Python), the callback should be careful to avoid direct
        modification of the context or state, which may lead to unexpected behavior
        or JAX tracer errors.

        Args:
            callback (Callable):
                The (pure) function to be wrapped. See above for expected signature.
            collect_inputs (bool):
                If True, the callback will eval input ports to gather input values.
                Normally this should be True, but it can be set to False for defining
                dummy "passthrough" callbacks for inactive branches of a call to
                `lax.cond`, for instance. This helps reduce the number of expressions
                that need to be JIT compiled. Default is True.

        Returns:
            Callable:
                The wrapped function, with signature `callback(context) -> result`.
        """

        # collect_inputs: if True, the callback will eval input ports to gather
        # input values.  Normally this should be True, but it can be set to False
        # for defining dummy "passthrough" callbacks for inactive branches of a
        # call to `lax.cond`, for instance. This helps reduce the number of expressions
        # that need to be JIT compiled.

        def _wrapped_callback(context: ContextBase) -> Array:
            if collect_inputs:
                inputs = self.collect_inputs(context)
            else:
                inputs = ()
            leaf_context: LeafContext = context[self.system_id]

            leaf_state = leaf_context.state
            params = leaf_context.parameters
            return callback(context.time, leaf_state, *inputs, **params)

        return _wrapped_callback

    @property
    def periodic_events(self) -> LeafCompositeEventCollection:
        return self._periodic_events

    @property
    def zero_crossing_events(self) -> LeafCompositeEventCollection:
        return self._zero_crossing_events

    # Inherits docstring from SystemBase
    def eval_discrete_updates(
        self, context, events: LeafCompositeEventCollection
    ) -> List[Array]:
        # Local copy of discrete state to avoid any in-place mutation
        xd = context[self.system_id].discrete_state.copy()
        local_events = events[self.system_id]

        logger.debug(f"Eval discrete events for {self.name}")
        logger.debug(f"xd: {xd}")
        logger.debug(f"local events: {local_events}")
        for event in local_events.discrete_update_events:
            # This is evaluated conditionally on event_data.active
            xd[event.state_index] = event.handle(context)

        return xd

    # Inherits docstring from SystemBase
    def eval_zero_crossing_updates(
        self,
        context: ContextBase,
        events: LeafCompositeEventCollection,
    ) -> LeafState:
        local_events = events[self.system_id]
        state = context[self.system_id].state

        logger.debug(f"Eval unrestricted events for {self.name}")
        logger.debug(f"local events: {local_events}")

        for event in local_events.unrestricted_update_events:
            # This is evaluated conditionally on event_data.active
            state = _handle_zero_crossing(self, event, context)

            # Store the updated state in the context for this block
            leaf_context = context[self.system_id].with_state(state)

            # Update the context for this block in the overall context
            context = context.with_subcontext(self.system_id, leaf_context)

        # Now `context` contains the updated "plus" state for this block, but
        # this needs to be discarded so that other block updates can also be
        # processed using the "minus" state. This is done by simply returning the
        # "plus" state and discarding the rest of the updated context.
        return state

    # Inherits docstring from SystemBase
    # TODO: This should be made obsolete following WC-120. Added as part of WC-80 to
    # remove the need for passing "handler" functions, which makes the code much
    # more straightforward.
    def eval_unrestricted_updates(
        self,
        context: ContextBase,
        events: LeafCompositeEventCollection,
    ) -> LeafState:
        local_events = events[self.system_id]
        state = context[self.system_id].state

        logger.debug(f"Eval unrestricted events for {self.name}")
        logger.debug(f"local events: {local_events}")

        for event in local_events.unrestricted_update_events:
            # This is evaluated conditionally on event_data.active
            state = _handle_unrestricted_update(self, event, context)

            # Store the updated state in the context for this block
            leaf_context = context[self.system_id].with_state(state)

            # Update the context for this block in the overall context
            context = context.with_subcontext(self.system_id, leaf_context)

        # Now `context` contains the updated "plus" state for this block, but
        # this needs to be discarded so that other block updates can also be
        # processed using the "minus" state. This is done by simply returning the
        # "plus" state and discarding the rest of the updated context.
        return state

    # Inherits docstring from SystemBase
    def determine_active_guards(
        self, root_context: ContextBase
    ) -> LeafCompositeEventCollection:
        # This can be overridden
        context = root_context[self.system_id]
        mode = context.mode  # Current system mode

        def _conditionally_activate(
            event: ZeroCrossingEvent,
        ) -> ZeroCrossingEvent:
            return cond(
                mode == event.active_mode,
                lambda e: e.mark_active(),
                lambda e: e.mark_inactive(),
                event,
            )

        zero_crossing_events = self.zero_crossing_events.mark_all_active()
        for i, event in enumerate(zero_crossing_events.unrestricted_update_events):
            # Check to see if the event corresponds to a mode transition
            if isinstance(event, ZeroCrossingEvent) and event.active_mode is not None:
                logger.debug(f"Mode: {mode}, {event.active_mode=}")
                # If the system is currently in the "start" mode, then the
                # event is active
                event = _conditionally_activate(event)

                # Store the updated event in the collection
                zero_crossing_events.unrestricted_update_events[i] = event

        logger.debug(f"Zero-crossing events for {self.name}: {zero_crossing_events}")
        return zero_crossing_events

    #
    # State/parameter declaration
    #
    def declare_continuous_state(
        self,
        shape: ShapeLike = None,
        default_value: Array = None,
        dtype: DTypeLike = None,
        ode: Callable = None,
        as_array: bool = True,
    ):
        """Declare a continuous state component for the system.

        The `ode` callback computes the time derivative of the continuous state based on the
        current time, state, and any additional inputs. If `ode` is not provided, a default
        zero vector of the same size as the continuous state is used. If provided, the `ode`
        callback should have the signature `ode(time, state, *inputs, **params) -> xcdot`.

        Args:
            shape (ShapeLike, optional):
                The shape of the continuous state vector. Defaults to None.
            default_value (Array, optional):
                The initial value of the continuous state vector. Defaults to None.
            dtype (DTypeLike, optional):
                The data type of the continuous state vector. Defaults to None.
            ode (Callable, optional):
                The callback for computing the time derivative of the continuous state.
                Should have the signature `ode(time, state, *inputs) -> xcdot`.
                Defaults to None.
            as_array (bool, optional):
                If True, treat the default_value as an array-like (cast if necessary).
                Otherwise, it will be stored as the default state without modification.

        Raises:
            AssertionError:
                If neither shape nor default_value is provided.

        Notes:
            (1) Only one of `shape` and `default_value` should be provided. If `default_value`
            is provided, it will be used as the initial value of the continuous state. If
            `shape` is provided, the initial value will be a zero vector of the given shape
            and specified dtype.
        """

        assert not (
            shape is None and default_value is None
        ), "Must provide either shape or default_value"
        if default_value is not None and as_array:
            default_value = array(default_value, dtype=dtype)
        else:
            default_value = zeros(shape, dtype=dtype)

        logger.debug(f"In block {self.name} [{self.system_id}]: {default_value=}")

        self._default_continuous_state = default_value

        if ode is None:
            # If no ODE is specified, return a zero vector of the same size as the
            # continuous state.
            def ode(time, state, *inputs, **parameters):
                return zeros_like(default_value)

        # Wrap the ode function to accept a context and return the time derivatives
        _ode_callback = self.wrap_update_callback(ode)

        # Create "cache entry" to store the callback and dependency info for the ODE.
        # Instead of having a new dependency ticket, this will be associated with
        # the existing "composite" xcdot ticket.  When there are multiple systems with
        # continuous states, this reflects the concept that the continuous state is shared
        # at the top level of the dynamical system (i.e. any continuous state might affect
        # any other continuous state).
        ode_cache_index = self.declare_cache(
            _ode_callback,
            name=f"{self.name}:xcdot",
            default_value=zeros_like(default_value),
            ticket=DependencyTicket.xcdot,
            prerequisites_of_calc=[DependencyTicket.all_sources],
        )  # Stores the callback in self.cache_sources[ode_cache_index]
        self._ode_cache = self.cache_sources[ode_cache_index]

        # Override the default `eval_time_derivatives` to use the cache evaluation
        self.eval_time_derivatives = self._ode_cache.eval

    def declare_discrete_state(
        self,
        shape: ShapeLike = None,
        default_value: Array = None,
        dtype: DTypeLike = None,
        as_array: bool = True,
    ) -> int:
        """Declare a new discrete state component for the system.

        The discrete state is a component of the system's state that can be updated
        at specific events, such as zero-crossings or periodic updates. Multiple
        discrete states can be declared, and each is associated with a unique index.
        The index is used to access and update the corresponding discrete state in
        the system's context during event handling.

        The declared discrete state is initialized with either the provided default
        value or zeros of the correct shape and dtype.

        Args:
            shape (ShapeLike, optional):
                The shape of the discrete state. Defaults to None.
            default_value (Array, optional):
                The initial value of the discrete state. Defaults to None.
            dtype (DTypeLike, optional):
                The data type of the discrete state. Defaults to None.
            as_array (bool, optional):
                If True, treat the default_value as an array-like (cast if necessary).
                Otherwise, it will be stored as the default state without modification.

        Returns:
            int:
                The index of the declared discrete state in the discrete state list.

        Raises:
            AssertionError:
                If as_array is True and neither shape nor default_value is provided.

        Notes:
            (1) Only one of `shape` and `default_value` should be provided. If
            `default_value` is provided, it will be used as the initial value of the
            continuous state. If `shape` is provided, the initial value will be a
            zero vector of the given shape and specified dtype.

            (2) Use `declare_periodic_discrete_update` to declare an update event that
            modifies a component of the discrete state at a recurring interval.
        """
        if as_array:
            assert not (
                shape is None and default_value is None
            ), "Must provide either shape or default_value"
            if default_value is not None:
                default_value = array(default_value, dtype=dtype)
            else:
                default_value = zeros(shape, dtype=dtype)

        index = len(self._default_discrete_state)
        self._default_discrete_state.append(default_value)

        assert index == len(self._discrete_state_tickets)
        self._discrete_state_tickets.append(next_dependency_ticket())

        return index

    def declare_parameter(
        self,
        name: str,
        default_value: Union[Array, InstanceParameter] = None,
        shape: ShapeLike = None,
        dtype: DTypeLike = None,
        as_array: bool = True,
    ):
        """Declare a numeric parameter for the system.

        Parameters are declared in the system and accessed through the context to
        maintain separation of data ownership. This method creates an entry in the
        system's default_parameters, recording the name, default value, and dependency
        ticket for later reference.

        The default value will be used to initialize the context, so it
        will also serve as the initial value unless explicitly overridden. In the
        simplest cases, parameters could be stored as attributes of the LeafSystem,
        but declaring them has the advantage of moving the values to the context,
        allowing them to be traced by JAX rather than stored as static data. This
        means they can be differentiated, vmapped, or otherwise modified without
        re-compiling the simulation.

        Args:
            name (str):
                The name of the parameter.
            default_value (Union[Array, InstanceParameter], optional):
                The default value of the parameter. InstanceParameters are used
                primarily internally for serialization and should not normally need
                to be used directly when implementing LeafSystems. Defaults to None.
            shape (ShapeLike, optional):
                The shape of the parameter. Defaults to None.
            dtype (DTypeLike, optional):
                The data type of the parameter. Defaults to None.
            as_array (bool, optional):
                If True, treat the default_value as an array-like (cast if necessary).
                Otherwise, it will be stored as the default state without modification.

        Raises:
            AssertionError:
                If the parameter with the given name is already declared.

        Notes:
            (1) Only one of `shape` and `default_value` should be provided. If
            `default_value` is provided, it will be used as the initial value of the
            continuous state. If `shape` is provided, the initial value will be a
            zero vector of the given shape and specified dtype.
        """
        assert (
            name not in self._default_parameters
        ), f"Parameter {name} already declared"

        if isinstance(default_value, InstanceParameter):
            # NOTE: we replace the string_value with the parameter name
            # This is to handle cases where a submodel has an instance parameter
            # eg. gain="g" (with `g` defined in the parent model), and the Gain
            # block in the submodel has gain="gain".
            # In this case we want to retain "gain" as the string value in the
            # Gain block.
            self._instance_parameters[name] = InstanceParameter(
                name=name,
                string_value=default_value.name,
                evaluated_value=default_value.evaluated_value,
            )
            default_value = default_value.evaluated_value
        elif isinstance(default_value, (np.ndarray, jnp.ndarray)):
            self._instance_parameters[name] = InstanceParameter(
                name=name,
                string_value=str(default_value.tolist()),
                evaluated_value=default_value,
            )
        else:
            self._instance_parameters[name] = InstanceParameter(
                name=name,
                string_value=str(default_value),
                evaluated_value=default_value,
            )

        assert not (
            shape is None and default_value is None
        ), "Must provide either shape or default_value"
        if default_value is not None:
            if as_array:
                default_value = array(default_value, dtype=dtype)
        else:
            default_value = zeros(shape, dtype=dtype)

        logger.debug(
            f"Adding parameter {name} to {self.name} with default: {default_value}"
        )

        self._default_parameters[name] = default_value
        self._parameter_tickets[name] = next_dependency_ticket()

    def declare_configuration_parameters(self, **params):
        """Declare a set of "configuration" parameters for the system.

        These parameters are non-numeric parameters used for block configuration.
        Their declaration as parameters rather than object attributes is mainly
        for the purpose of serialization - blocks that take boolean or string
        parameters can register them as configuration parameters and they will be
        properly serialized.

        The args should be a dict of name-value pairs, where the values are either
        strings, bool, arrays, or InstanceParameters.

        Typical usage:

        ```python
        class MyBlock(LeafSystem):
            def __init__(self, param1=True, param2=1.0):
                super().__init__()
                self.declare_configuration_parameters(param1=param1, param2=param2)
        ```
        """
        for name, value in params.items():
            if isinstance(value, InstanceParameter):
                # NOTE: we replace the string_value with the parameter name
                # This is to handle cases where a submodel has an instance parameter
                # gain="g", and the Gain block in the submodel has gain="gain".
                # In this case we want to retain "gain" as the string value in the
                # Gain block.
                self._instance_parameters[name] = InstanceParameter(
                    name=name,
                    string_value=value.name,
                    evaluated_value=value.evaluated_value,
                )
            elif isinstance(value, (np.ndarray, jnp.ndarray)):
                self._instance_parameters[name] = InstanceParameter(
                    name=name,
                    string_value=str(value.tolist()),
                    evaluated_value=value,
                )
            else:
                self._instance_parameters[name] = InstanceParameter(
                    name=name,
                    string_value=str(value),
                    evaluated_value=value,
                )

    #
    # I/O declaration
    #
    def declare_continuous_state_output(
        self,
        name: str = None,
    ) -> int:
        """Declare a continuous state output port in the system.

        This method creates a new block-level output port which returns the full
        continuous state of the system.

        Args:
            name (str, optional):
                The name of the output port. Defaults to None (autogenerate name).

        Returns:
            int: The index of the new output port.
        """

        def _callback(context: ContextBase):
            return context[self.system_id].continuous_state

        return self.declare_output_port(
            _callback,
            name=name,
            prerequisites_of_calc=[DependencyTicket.xc],
            default_value=self._default_continuous_state,
        )

    def declare_discrete_state_output(
        self,
        state_index: int = 0,
        name: str = None,
    ) -> int:
        """Declare a discrete state output port in the system.

        This method creates a new block-level output port which returns the component
        of the system's discrete state at the given index.

        Args:
            state_index (int, optional):
                The index of the discrete state component to output. Defaults to 0.
            name (str, optional):
                The name of the output port. Defaults to None (autogenerate name).

        Returns:
            int: The index of the new output port.
        """

        def _callback(context: ContextBase):
            return context[self.system_id].discrete_state[state_index]

        return self.declare_output_port(
            _callback,
            name=name,
            prerequisites_of_calc=[DependencyTicket.xd],
            default_value=self._default_discrete_state[state_index],
        )

    def declare_mode_output(self, name: str = None) -> int:
        """Declare a mode output port in the system.

        This method creates a new block-level output port which returns the component
        of the system's state corresponding to the discrete "mode" or "stage".

        Args:
            name (str, optional):
                The name of the output port. Defaults to None.

        Returns:
            int:
                The index of the declared mode output port.
        """

        def _callback(context: ContextBase):
            return context[self.system_id].mode

        return self.declare_output_port(
            _callback,
            name=name,
            prerequisites_of_calc=[DependencyTicket.mode],
            default_value=self._default_mode,
        )

    #
    # Event declaration
    #
    def declare_periodic_discrete_update(
        self,
        callback: Callable,
        period: Scalar,
        offset: Scalar,
        state_index: int = 0,
        enable_tracing: bool = None,
    ):
        """Declare a periodic discrete update event.

        The event will be triggered at regular intervals defined by the period and
        offset parameters. The callback should have the signature
        `callback(time, state, *inputs, **params) -> xd_plus`, where `xd_plus` is the
        updated value of the discrete state component at the given index.

        This callback should be written to compute the "plus" value of the discrete
        state component given the "minus" values of all state components and inputs.

        Args:
            callback (Callable):
                The callback function defining the update.
            period (Scalar):
                The period at which the update event occurs.
            offset (Scalar):
                The offset at which the first occurrence of the event is triggered.
            state_index (int, optional):
                The index of the discrete state to be updated. Defaults to 0.
            enable_tracing (bool, optional):
                If True, enable tracing for this event. Defaults to None.
        """

        # Dummy callback when the event is inactive. This is necessary because the
        # event collection must have a consistent size throughout simulation. With the
        # dummy callback events can be "deactivated" and kept in the event collection.
        def _passthrough(
            time: Scalar, state: LeafState, *inputs, **parameters
        ) -> Array:
            return state.discrete_state[state_index]

        if enable_tracing is None:
            enable_tracing = self.enable_trace_discrete_updates
        logger.debug(
            f"Declaring periodic discrete update for {self.name} with callback"
            f" {callback}: traced? {enable_tracing}"
        )
        event = DiscreteUpdateEvent(
            event_data=PeriodicEventData(period=period, offset=offset, active=False),
            callback=self.wrap_update_callback(callback),
            passthrough=self.wrap_update_callback(_passthrough, collect_inputs=False),
            state_index=state_index,
            enable_tracing=enable_tracing,
        )
        self._periodic_events.add_discrete_update_event(event)

    def declare_periodic_unrestricted_update(
        self,
        callback: Callable,
        period: Scalar,
        offset: Scalar,
        enable_tracing: bool = None,
    ):
        """Declare a periodic unrestricted (full-state) update event.

        The event will be triggered at regular intervals defined by the period and
        offset parameters. The callback should have the signature
        `callback(time, state, *inputs, **params) -> x_plus`, where `x_plus` is the
        updated state of the system (an entire LeafState).

        This callback should be written to compute the "plus" state given the "minus"
        values of all state components and inputs.  Since the state is immutable, the
        updated state returned from the callback will necessarily be a copy.

        Args:
            callback (Callable):
                The callback function defining the update.
            period (Scalar):
                The period at which the update event occurs.
            offset (Scalar):
                The offset at which the first occurrence of the event is triggered.
            state_index (int, optional):
                The index of the discrete state to be updated. Defaults to 0.
            enable_tracing (bool, optional):
                If True, enable tracing for this event. Defaults to None.
        """

        # Dummy callback when the event is inactive. This is necessary because the
        # event collection must have a consistent size throughout simulation. With the
        # dummy callback events can be "deactivated" and kept in the event collection.
        def _passthrough(
            time: Scalar, state: LeafState, *inputs, **parameters
        ) -> LeafState:
            return state

        if enable_tracing is None:
            enable_tracing = self.enable_trace_unrestricted_updates
        event = UnrestrictedUpdateEvent(
            event_data=PeriodicEventData(period=period, offset=offset, active=False),
            callback=self.wrap_update_callback(callback),
            passthrough=self.wrap_update_callback(_passthrough, collect_inputs=False),
            enable_tracing=enable_tracing,
        )
        self._periodic_events.add_unrestricted_update_event(event)

    # TODO: Deprecate publish events
    # (following https://github.com/collimator-ai/collimator/pull/5010)
    def declare_periodic_publish(self, period, offset, callback):
        # Here the callback may need to use the context directly, so don't wrap it
        event = PublishEvent(
            callback=callback,
            event_data=PeriodicEventData(period=period, offset=offset, active=False),
        )
        self._periodic_events.add_publish_event(event)

    def declare_default_mode(self, mode: int):
        self._default_mode = mode

    def declare_zero_crossing(
        self,
        guard: Callable,
        reset_map: Callable = None,
        start_mode: int = None,
        end_mode: int = None,
        direction: str = "crosses_zero",
        name: str = None,
        enable_tracing: bool = None,
    ):
        """Declare an event triggered by a zero-crossing of a guard function.

        Optionally, the system can also transition between discrete modes
        If `start_mode` and `end_mode` are specified, the system will transition
        from `start_mode` to `end_mode` when the event is triggered according to `guard`.
        This event will be active conditionally on `state.mode == start_mode` and when
        triggered will result in applying the reset map. In addition, the mode will be
        updated to `end_mode`.

        If `start_mode` and `end_mode` are not specified, the event will always be active
        and will not result in a mode transition.

        The guard function should have the signature:
            `guard(time, state, *inputs, **parameters) -> float`

        and the reset map should have the signature of an unrestricted update:
            `reset_map(time, state, *inputs, **parameters) -> state`

        Args:
            guard (Callable):
                The guard function which triggers updates on zero crossing.
            reset_map (Callable, optional):
                The reset map which is applied when the event is triggered. If None
                (default), no reset is applied.
            start_mode (int, optional):
                The mode or stage of the system in which the guard will be
                actively monitored. If None (default), the event will always be
                active.
            end_mode (int, optional):
                The mode or stage of the system to which the system will transition
                when the event is triggered. If start_mode is None, this is ignored.
                Otherwise it _must_ be specified, though it can be the same as
                start_mode.
            direction (str, optional):
                The direction of the zero crossing. Options are "crosses_zero"
                (default), "positive_then_non_positive", "negative_then_non_negative",
                and "edge_detection".  All except edge detection operate on continuous
                signals; edge detection operates on boolean signals and looks for a
                jump from False to True or vice versa.
            name (str, optional):
                The name of the event. Defaults to None.
            enable_tracing (bool, optional):
                If True, enable tracing for this event. Defaults to None.

        Notes:
            By default the system state does not have a "mode" component, so in
            order to declare "state transitions" with non-null start and end modes,
            the user must first call `declare_default_mode` to set the default mode
            to be some integer (initial condition for the system).
        """

        logger.debug(
            f"Declaring transition for {self.name} with guard {guard} and reset map {reset_map}"
        )

        if start_mode is not None or end_mode is not None:
            assert (
                self._default_mode is not None
            ), "System has no mode: call `declare_default_mode` before transitions."
            assert isinstance(start_mode, int) and isinstance(end_mode, int)

        # Default behavior of enable_tracing is to inherit from the class variable.
        if enable_tracing is None:
            enable_tracing = self.enable_trace_unrestricted_updates

        # Dummy callback when the event is inactive. This is necessary because the
        # event collection must have a consistent size throughout simulation. With the
        # dummy callback events can be "deactivated" and kept in the event collection.
        def _passthrough(
            time: Scalar, state: LeafState, *inputs, **parameters
        ) -> LeafState:
            return state

        # Wrap the reset map with a mode update if necessary
        def _reset_and_update_mode(
            time: Scalar, state: LeafState, *inputs, **parameters
        ) -> LeafState:
            if reset_map is not None:
                state = reset_map(time, state, *inputs, **parameters)
            logger.debug(f"Updating mode from {state.mode} to {end_mode}")

            # If the start and end modes are declared, update the mode
            if start_mode is not None:
                logger.debug(f"Updating mode from {state.mode} to {end_mode}")
                state = state.with_mode(end_mode)

            return state

        event = ZeroCrossingEvent(
            guard=self.wrap_update_callback(guard),
            reset_map=self.wrap_update_callback(_reset_and_update_mode),
            passthrough=self.wrap_update_callback(_passthrough, collect_inputs=False),
            direction=direction,
            name=name,
            event_data=ZeroCrossingEventData(active=True),
            enable_tracing=enable_tracing,
            active_mode=start_mode,
        )

        event_index = self._zero_crossing_events.num_unrestricted_update_events
        self._zero_crossing_events.add_unrestricted_update_event(event)

        # Record the transition in the transition map (for debugging or analysis)
        if start_mode is not None:
            if start_mode not in self.transition_map:
                self.transition_map[start_mode] = []
            self.transition_map[start_mode].append((event_index, event))

    #
    # Initialization
    #
    @property
    def context_factory(self) -> LeafContextFactory:
        return LeafContextFactory(self)

    @property
    def dependency_graph_factory(self) -> LeafDependencyGraphFactory:
        return LeafDependencyGraphFactory(self)

    def create_state(self) -> LeafState:
        # Hook for context creation: get the default state for this system.
        # Users should not need to call this directly - the state will be created
        # as part of the context.  Generally, `system.create_context()` should
        # be all that's necessary for initialization.

        return LeafState(
            name=self.name,
            continuous_state=self._default_continuous_state,
            discrete_state=self._default_discrete_state,
            mode=self._default_mode,
        )

    def create_parameters(self) -> dict[str, Array]:
        # Hook for context creation: get the default parameters for this system.
        # Users should not need to call this directly - the parameters will be created
        # as part of the context.  Generally, `system.create_context()` should
        # be all that's necessary for initialization.
        return self._default_parameters.copy()

    # Inherits docstring from SystemBase.get_feedthrough
    def get_feedthrough(self) -> List[Tuple[int, int]]:
        # NOTE: This implementation is basically a direct port of the Drake algorithm

        # If we already did this or it was set manually, return the stored value
        if self.feedthrough_pairs is not None:
            return self.feedthrough_pairs

        feedthrough = []  # Confirmed feedthrough pairs (input, output)

        # First collect all possible feedthrough pairs
        unknown: Set[Tuple[int, int]] = set()
        for iport in self.input_ports:
            for oport in self.output_ports:
                unknown.add((iport.index, oport.index))

        if len(unknown) == 0:
            return feedthrough

        # Create a local context and cache.
        # This cache will only contain local sources - this is fine since we're just
        # testing local input -> output paths for this system. Since the context is
        # local, it may have disconnected inputs, so automatic type checking must be
        # skipped
        #
        # TODO: The context is only needed for the cache: separate these and
        #  create a separate minimal "cache" for feedthrough testing
        cache = self.create_context(check_types=False).cache

        original_unknown = unknown.copy()
        for pair in original_unknown:
            u, v = pair
            output_port = self.output_ports[v]
            input_port = self.input_ports[u]

            # If output prerequisites are unspecified, this tells us nothing
            if output_port.has_default_prerequisites:
                continue

            # Determine feedthrough dependency via cache invalidation
            cache = _mark_up_to_date(cache, output_port.cache_index)

            # Notify subscribers of a value change in the input, invalidating all
            # downstream cache values
            input_tracker = self.dependency_graph[input_port.ticket]
            cache = input_tracker.notify_subscribers(cache, self.dependency_graph)

            # If the output cache is now out of date, this is a feedthrough path
            if cache[output_port.cache_index].is_out_of_date:
                feedthrough.append(pair)

            # Regardless of the result of the caching, the pair is no longer unknown
            unknown.remove(pair)

            # Reset the output cache to out-of-date in case other inputs also
            # feed through to this output.
            cache = _mark_out_of_date(cache, output_port.cache_index)

        logger.debug(f"{self.name} feedthrough pairs: {feedthrough}")

        # Conservatively assume everything still unknown is feedthrough
        for pair in unknown:
            feedthrough.append(pair)

        self.feedthrough_pairs = feedthrough
        return self.feedthrough_pairs


#
# Event handling with custom adjoint definitions
#
# These are defined outside of the SystemBase class so that custom VJPs can be defined
# to handle "saltation" effects in zero-crossing events.
def _handle_unrestricted_update(
    _system: SystemBase, event: UnrestrictedUpdateEvent, context: ContextBase
):
    return event.handle(context)


# Specialized function with custom VJP for time-dependent state transitions
# with guards, resets, etc.
_handle_zero_crossing = jax.custom_vjp(
    _handle_unrestricted_update, nondiff_argnums=(0,)
)
_handle_zero_crossing.__annotations__ = _handle_unrestricted_update.__annotations__


def _handle_zero_crossing_fwd(
    system: SystemBase, event: ZeroCrossingEvent, context: ContextBase
) -> tuple[LeafState, tuple]:
    """Compute the "forward pass" of the zero-crossing update function.

    This basically just wraps the event handler function, but it also computes various
    "residual" information that will be necessary for the backwards pass. The way to
    understand what residuals are needed is to start with the adjoint function and then
    see what information used there can be more efficiently computed in the forward
    pass.
    """
    guard = event.eval_guard
    system_id = system.system_id

    # Evaluate the ODE RHS function.  This is needed to get the time sensitivity
    # of the continuous state, which is used in the adjoint update to time.
    def _ode(x: LeafState, context: ContextBase) -> Array:
        local_context = context[system_id].with_state(x)
        context = context.with_subcontext(system_id, local_context)
        return system.eval_time_derivatives(context)

    # Evaluate the _local_ system dynamics before the transition
    x_minus = context[system_id].state
    logger.debug(f"_handle_zero_crossing_fwd: {context.state=}")
    xdot_minus = _ode(x_minus, context)  # This is the _local_ xdot
    logger.debug(f"_handle_zero_crossing_fwd: {xdot_minus=}")

    # Evaluate the reset map
    def _reset_map(context) -> LeafState:
        return event.handle(context)

    # Evaluate the _local_ gradients of the guard with x_minus
    root_context_adj: ContextBase = jax.grad(guard, allow_int=True)(context)
    context_adj = root_context_adj[system_id]
    dg_dx = context_adj.continuous_state
    dg_dp = context_adj.parameters
    dg_dt = root_context_adj.time  # time is only defined at the root

    # Reset map VJP, and compute the transition (primal values)
    # Returns the _local_ updated state x_plus, and the function to compute
    # the vjp of the reset map given the adjoint state. This will be used in the
    # backwards pass.
    x_plus, reset_vjp = jax.vjp(_reset_map, context)

    # Recompute the _local_ ODE values after the transition
    xdot_plus = _ode(x_plus, context)

    # Combine all the "residuals" necessary for the backwards pass into a tuple.
    res = (dg_dx, dg_dt, dg_dp, xdot_minus, xdot_plus, reset_vjp)
    return x_plus, res


def _handle_zero_crossing_adj(
    system: SystemBase, res: tuple, state_adj: LeafState
) -> ContextBase:
    """Compute the "backward pass" of the zero-crossing update function.

    Ignoring "saltation" effects, we could just use the built-in autodiff.
    However, we need to correct for variations in the "time of impact", which
    has ramifications for the continuous state, time, and parameters in the
    _local_ context.  Only the local context is affected since we assume that the
    guard and reset maps can only see and act on the local state. We need to
    compute these corrections to the adjoints and then store them in an updated
    _root_ context (since this was the original input).

    The adjoint update to time is also overridden at the Simulator level
    because it is most naturally defined as t_adj = dot(xf_dot, vf) at the
    end of the `advance` call. However, we also need to update it here in case the
    adjoint time variable is used by other things farther in the graph.

    For a helpful tutorial related to this implementation, see:
    "Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems"
    https://arxiv.org/abs/2306.06862
    This is a bit more complicated because we allow for differentiation with respect to
    more than just the initial state, but it introduces the "saltation" idea.
    """
    system_id = system.system_id

    # Unpack the residuals from the forward pass
    dg_dx, dg_dt, dg_dp, xdot_minus, xdot_plus, reset_vjp = res

    # Compute vjp with reset Jacobian (return is a tuple of length 1). If we did not
    # account for saltation effects we could just return this directly.
    root_context_adj: ContextBase = reset_vjp(state_adj)[0]

    # Unpack the adjoint variables associated with the reset map
    context_adj = root_context_adj[system_id]
    vT_dR_dx = context_adj.continuous_state
    vT_dR_dp = context_adj.parameters
    vT_dR_dt = root_context_adj.time  # Time is only defined at the root

    context_adj = context_adj.with_state(state_adj)

    # Get the adjoint continuous state
    vc = state_adj.continuous_state

    # Compute vjp with rank-1 correction
    den = dg_dt + jnp.dot(dg_dx, xdot_minus)
    num = jnp.dot(xdot_plus, vc) - jnp.dot(vT_dR_dx, xdot_minus) - vT_dR_dt
    gamma = jnp.where(den != 0, num / den, 0.0 * num)
    vT_C = gamma * dg_dx  # Correction to vjp with the reset map

    # Full vjp with saltation matrix (corrected reset vjp)
    vT_Xi = vT_dR_dx + vT_C

    # Update context with the adjoint variables associated with the continuous state
    context_adj = context_adj.with_continuous_state(vT_Xi)

    # Adjoint update to the parameters: vT * (dR_dp + frac * dg_dp)
    vT_dp = jax.tree_map(lambda x, y: x + gamma * y, vT_dR_dp, dg_dp)

    # Update the adjoint variables associated with the parameters
    context_adj = context_adj.with_parameters(vT_dp)

    # Update the adjoint variables associated with time
    t_adj = vT_dR_dt + gamma * dg_dt

    root_context_adj = root_context_adj.with_subcontext(system_id, context_adj)
    root_context_adj = root_context_adj.with_time(t_adj)

    event_adj = None  # No adjoint needed for the event itself
    return event_adj, root_context_adj


_handle_zero_crossing.defvjp(_handle_zero_crossing_fwd, _handle_zero_crossing_adj)
