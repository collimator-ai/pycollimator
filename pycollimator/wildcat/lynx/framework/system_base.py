"""Base class for System objects.

Systems are the basic building blocks of a model in the wildcat framework.
In isolation they represent a hybrid dynamical system, but they can also be
combined into block diagrams to represent more complex models.  The interface
defined by SystemBase is intended for composability, so diagrams can themselves
be treated as "blocks" in larger diagrams.

Systems are intended to act as a collection of pure functions, with the
"dynamic" data (time, state, etc.) stored in a corresponding Context object.
For example, `system.eval_time_derivatives(context)` will return the time
derivatives of the continuous state for the system, given the root context of
the diagram.
"""
from __future__ import annotations

import abc
from typing import (
    List,
    Tuple,
    TYPE_CHECKING,
    Hashable,
    Any,
    Callable,
    ClassVar,
)
import dataclasses

import jax.numpy as jnp
import numpy as np

from ..logging import logger
from .cache import CacheSource
from .port import InputPort, OutputPort
from .dependency_graph import DependencyTicket
from .error import StaticError

__all__ = [
    "SystemBase",
    "InstanceParameter",
]

if TYPE_CHECKING:
    # Array-like object, e.g. a JAX PyTree or a numpy ndarray.
    from ..math_backend.typing import Array

    from .context import ContextBase
    from .context_factory import ContextFactory
    from .diagram import Diagram
    from .event import CompositeEventCollection
    from .port import InputPortLocator, PortBase
    from .cache import Cache
    from .dependency_graph import DependencyGraph, DependencyGraphFactory

    # See note on "state" type hints in the docstring of SystemBase.
    from .state import State, ContinuousState, DiscreteState


def next_system_id() -> int:
    """Get the next unique system ID."""
    return IDGenerator.get_next_id()


def reset_system_id():
    """Reset the system ID counter."""
    IDGenerator.reset()


class IDGenerator:
    """Singleton class for automatically managing unique system IDs.

    This can be ignored altogether if using manually determined system IDs,
    for example as determined by "block path" from the UI. In either case, it
    typically does not need to be used directly by either users or developers.
    """

    _id = 0

    @classmethod
    def get_next_id(cls):
        cls._id += 1
        return cls._id

    @classmethod
    def reset(cls):
        cls._id = 0


class UpstreamEvalError(StaticError):
    def __init__(self, input_port_locator: InputPortLocator):
        system, port_index = input_port_locator
        msg = (
            f"Evaluation of input port {system.name}[{port_index}] failed - "
            "may be disconnected or part of a subsystem."
        )
        super().__init__(system.system_id, msg)


@dataclasses.dataclass
class InstanceParameter:
    """This class is used during import of a model from the dashboard to preserve
    the string representation of a parameter and its evaluated value.
    """

    name: str
    string_value: str
    evaluated_value: Any

    def _op(self, op_char: str, op_fn: Callable, other):
        if isinstance(other, InstanceParameter):
            return InstanceParameter(
                name=f"{self.name} {op_char} {other.name}",
                string_value=f"{self.name} {op_char} {other.name}",
                evaluated_value=op_fn(self.evaluated_value, other.evaluated_value),
            )

        other_str = str(other)
        if isinstance(other, (np.ndarray, jnp.ndarray)):
            other_str = f"{other.tolist()}"
        return InstanceParameter(
            name=f"{self.name} {op_char} {other_str}",
            string_value=f"{self.string_value} {op_char} {other_str}",
            evaluated_value=op_fn(self.evaluated_value, other),
        )

    def __add__(self, other):
        return self._op("+", lambda x, y: x + y, other)

    def __sub__(self, other):
        return self._op("-", lambda x, y: x - y, other)

    def __mul__(self, other):
        return self._op("*", lambda x, y: x * y, other)

    def __truediv__(self, other):
        return self._op("/", lambda x, y: x / y, other)

    def __floordiv__(self, other):
        return self._op("//", lambda x, y: x // y, other)

    def __mod__(self, other):
        return self._op("%", lambda x, y: x % y, other)

    def __pow__(self, other):
        return self._op("**", lambda x, y: x**y, other)

    def __neg__(self):
        return InstanceParameter(
            name=f"-{self.name}",
            string_value=f"-{self.string_value}",
            evaluated_value=-self.evaluated_value,
        )

    def __pos__(self):
        return InstanceParameter(
            name=f"+{self.name}",
            string_value=f"+{self.string_value}",
            evaluated_value=+self.evaluated_value,
        )

    def __abs__(self):
        return InstanceParameter(
            name=f"abs({self.name})",
            string_value=f"abs({self.string_value})",
            evaluated_value=abs(self.evaluated_value),
        )

    def __eq__(self, other):
        return self._op("==", lambda x, y: x == y, other)

    def __ne__(self, other):
        return self._op("!=", lambda x, y: x != y, other)

    def __lt__(self, other):
        return self._op("<", lambda x, y: x < y, other)

    def __le__(self, other):
        return self._op("<=", lambda x, y: x <= y, other)

    def __gt__(self, other):
        return self._op(">", lambda x, y: x > y, other)

    def __ge__(self, other):
        return self._op(">=", lambda x, y: x >= y, other)


@dataclasses.dataclass
class SystemBase:
    """Basic building block for simulation in wildcat.

    NOTE: Type hints in SystemBase indicate the union between what would be returned
    by a LeafSystem and a Diagram. See type hints of the subclasses for the specific
    argument and return types.
    """

    system_id: Hashable = None  # Unique ID for this system
    name: str = None  # Human-readable name for this system

    # Flags to enable/disable JAX tracing of various functions
    # defined on the system.
    enable_trace_cache_sources: ClassVar[bool] = True
    enable_trace_time_derivatives: ClassVar[bool] = True
    enable_trace_discrete_updates: ClassVar[bool] = True
    enable_trace_unrestricted_updates: ClassVar[bool] = True

    # Immediate parent of the current system (can only be a Diagram).
    # If None, _this_ is the root system.
    parent: Diagram = None

    def __post_init__(self):
        if self.system_id is None:
            self.system_id = next_system_id()

        if self.name is None:
            self.name = f"{type(self).__name__}_{self.system_id}_"

        # All "cache sources" for this system. Typically these will correspond to
        # input ports, output ports, time derivative calculations, and any custom
        # "cached data" declared by the user (e.g. see ModelicaFMU block).
        self.cache_sources: List[CacheSource] = []

        # Index into cache_sources for each port. For instance, input port `i` can
        # be retrieved by `self.cache_sources[self.input_port_indices[i]]`. The
        # `input_ports` and `output_ports` properties give more convenient access.
        self.input_port_indices: List[int] = []
        self.output_port_indices: List[int] = []

        # Override this or set manually to provide a custom characteristic time scale
        # for the system. At the moment this is only used for zero-crossing isolation
        # in the simulator.
        self.characteristic_time = 1.0

        # A dependency graph for the system, mapping prerequisites of each calculation.
        # `None` indicates that the dependency graph has not been constructed yet.  If
        # accessed via the `dependency_graph` property, it will be constructed
        # automatically as necessary.
        self._dependency_graph: DependencyGraph = None

        # Storage of attributes for system serialization
        self._instance_parameters: dict[str, InstanceParameter] = {}

        # Map from (input_port, output_port) if that pair is feedthrough
        # `None` indicates that the feedthrough is unknown for this system.
        # This will be computed automatically using the dependency graph
        # during algebraic loop detection unless it is set manually.
        # To manually set feedthrough, either declare this explicitly or
        # override `get_feedthrough`.
        self.feedthrough_pairs: List[Tuple[int, int]] = None

    def __hash__(self) -> Hashable:
        return hash(self.system_id)

    def pprint(self, output=print) -> str:
        output(self._pprint_helper().strip())

    def _pprint_helper(self, prefix="") -> str:
        return f"{prefix}|-- {self.name}(id={self.system_id})\n"

    #
    # Serialization
    #
    @property
    def default_parameters(self) -> dict[str, Array]:
        return {}

    @property
    def instance_parameters(self) -> dict[str, InstanceParameter]:
        return self._instance_parameters

    @instance_parameters.setter
    def instance_parameters(self, value):
        self._instance_parameters = value

    #
    # Simulation interface
    #
    def eval_time_derivatives(self, context: ContextBase) -> ContinuousState:
        """Evaluate the continuous time derivatives for this system.

        Given the _root_ context, evaluate the continuous time derivatives,
        which must have the same PyTree structure as the continuous state.

        In principle, this can be overridden by custom implementations, but
        in general it is preferable to declare continuous states for LeafSystems
        using `declare_continuous_state`, which accepts a callback function
        that will be used to compute the derivatives. For Diagrams, the time
        derivatives are computed automatically using the callback functions for
        all child systems with continuous state.

        Args:
            context (ContextBase): root context of this system

        Returns:
            PyTree: continuous time derivatives for this system
        """
        return None

    def eval_discrete_updates(
        self, context: ContextBase, events: CompositeEventCollection
    ) -> DiscreteState:
        """Compute updates to the discrete states of the system.

        Given the _root_ context, evaluate the discrete updates, which must have the
        same PyTree structure as the discrete states of this system. This should be
        a pure function, so that it does not modify any aspect of the context in-place
        (even though it is difficult to strictly prevent this in Python). These might
        be triggered periodically or by zero-crossing events.

        Following the specifications in Drake, all updates should be computed based
        on the `xd_minus` values (those in the context), and should not attempt to use
        any `xd_plus` values from other calculations, even if it is somehow possible to
        access them.  See:

        https://drake.mit.edu/doxygen_cxx/group__discrete__systems.html

        Although this can technically be directly overridden by custom implementations,
        in general it is preferable to declare discrete states for LeafSystems using
        `declare_discrete_state` and define updates using methods like
        `declare_discrete_periodic_update`, which will automatically create dependency
        tickets, add events to the system's event collection, and extend the system's
        discrete state component properly. For Diagrams, the discrete updates are
        computed automatically using the callback functions for all child systems with
        discrete state.

        Args:
            context (ContextBase): root context of this system with `xd_minus` values.
            events (CompositeEventCollection): set of discrete event updates
                which may be applied.  These events contain data which tracks their
                current active/inactive state, so they may not actually all be applied.

        Returns:
            DiscreteState: updated discrete states `xd_plus`

        Notes:
            The events are evaluated conditionally on being marked "active", so
            the entire event collection can be passed without filtering to active events.
            This is necessary to make the function calls work with JAX tracing, which do
            not allow for variable-sized arguments or returns.
        """
        return 0  # return 0 makes pylint happy

    @abc.abstractmethod
    def eval_zero_crossing_updates(
        self,
        context: ContextBase,
        events: CompositeEventCollection,
    ) -> State:
        """Evaluate reset maps associated with zero-crossing events.

        Args:
            context (ContextBase):
                The context for the system, containing the current state and parameters.
            events (CompositeEventCollection):
                The collection of events to be evaluated (for example zero-crossing or
                periodic events for this system).

        Returns:
            State:
                The complete state with all updates applied.

        Notes:
            (1) Following the Drake definition, "unrestricted" updates are allowed to
            modify any component of the state: continuous, discrete, or mode.  These
            updates are evaluated in the order in which they were declared, so it is
            _possible_ (but should be strictly avoided) for multiple events to modify the
            same state component at the same time.

            Each update computes its results given the _current_ state of the system
            (the "minus" values) and returns the _updated_ state (the "plus" values).
            The update functions cannot access any information about the "plus" values of
            its own state or the state of any other block.  This could change in the future
            but for now it ensures consistency with Drake's discrete semantices:

            More specifically, since all unrestricted updates can modify the entire state,
            any time there are multiple unrestricted updates, the resulting states are
            ALWAYS in conflict.  For example, suppose a system has two unrestricted
            updates, `event1` and `event2`.  At time t_n, `event1` is active and `event2`
            is inactive.  First, `event1` is evaluated, and the state is updated.  Then
            `event2` is evaluated, but the state is not updated.  Which one is valid?
            Obviously, the `event1` return is valid, but how do we communicate this to JAX?
            The situation is more complicated if both `event1` and `event2` happen to be
            active.  In this case the states have to be "merged" somehow.  In the worst
            case, these two will modify the same components of the state in different ways.

            The implementation updates the state in a local copy of the context (since both
            are immutable).  This allows multiple unrestricted updates, but leaves open the
            possibility of multiple active updates modifying the state in conflicting ways.
            This should strictly be avoided by the implementer of the LeafSystem.  If it is
            at all unclear how to do this, it may be better to split the system into
            multiple blocks to be safe.

            (2) The events are evaluated conditionally on being marked "active"
            (indicating that their guard function triggered), so the entire event
            collection can be passed without filtering to active events. This is necessary
            to make the function calls work with JAX tracing, which do not allow for
            variable-sized arguments or returns.
        """
        pass

    @abc.abstractmethod
    def eval_unrestricted_updates(
        self,
        context: ContextBase,
        events: CompositeEventCollection,
    ) -> State:
        """Evaluate unrestricted update functions in the given event collection.

        TODO: This is essentially a duplicate of `eval_zero_crossing_updates` without
        the custom VJP rule. It should be made obsolete with the removal of periodic
        unrestricted updates. See:
        https://collimator.atlassian.net/browse/WC-120

        Args:
            handler (Callable):
                The update handler function, which should have the signature
                `handler(system, event, context) -> state`. This will be one of the
                handler functions defined in `wildcat.simulation.event_handling`.
            context (ContextBase):
                The context for the system, containing the current state and parameters.
            events (CompositeEventCollection):
                The collection of events to be evaluated (for example zero-crossing or
                periodic events for this system).

        Returns:
            State:
                The complete state with all updates applied.

        Notes:
            (1) Following the Drake definition, "unrestricted" updates are allowed to
            modify any component of the state: continuous, discrete, or mode.  These
            updates are evaluated in the order in which they were declared, so it is
            _possible_ (but should be strictly avoided) for multiple events to modify the
            same state component at the same time.

            Each update computes its results given the _current_ state of the system
            (the "minus" values) and returns the _updated_ state (the "plus" values).
            The update functions cannot access any information about the "plus" values of
            its own state or the state of any other block.  This could change in the future
            but for now it ensures consistency with Drake's discrete semantices:

            More specifically, since all unrestricted updates can modify the entire state,
            any time there are multiple unrestricted updates, the resulting states are
            ALWAYS in conflict.  For example, suppose a system has two unrestricted
            updates, `event1` and `event2`.  At time t_n, `event1` is active and `event2`
            is inactive.  First, `event1` is evaluated, and the state is updated.  Then
            `event2` is evaluated, but the state is not updated.  Which one is valid?
            Obviously, the `event1` return is valid, but how do we communicate this to JAX?
            The situation is more complicated if both `event1` and `event2` happen to be
            active.  In this case the states have to be "merged" somehow.  In the worst
            case, these two will modify the same components of the state in different ways.

            The implementation updates the state in a local copy of the context (since both
            are immutable).  This allows multiple unrestricted updates, but leaves open the
            possibility of multiple active updates modifying the state in conflicting ways.
            This should strictly be avoided by the implementer of the LeafSystem.  If it is
            at all unclear how to do this, it may be better to split the system into
            multiple blocks to be safe.

            (2) The events are evaluated conditionally on being marked "active"
            (indicating that their guard function triggered), so the entire event
            collection can be passed without filtering to active events. This is necessary
            to make the function calls work with JAX tracing, which do not allow for
            variable-sized arguments or returns.
        """
        pass

    def handle_discrete_update(
        self, events: CompositeEventCollection, context: ContextBase
    ) -> ContextBase:
        """Compute and apply active discrete updates.

        This is intended for internal use by the simulator and should not normally need
        to be invoked directly by users. Events are evaluated conditionally on being
        marked "active", so the entire event collection can be passed without filtering
        to active events. This is necessary to make the function calls work with JAX
        tracing, which do not allow for variable-sized arguments or returns.

        Args:
            events (CompositeEventCollection): collection of discrete update events
            context (ContextBase): root context for this system

        Returns:
            ContextBase:
                updated context with all active updates applied to the discrete state
        """
        logger.debug(
            f"Handling {events.num_discrete_update_events} discrete update events at t={context.time}"
        )
        if events.num_discrete_update_events > 0:
            # For some reason pylint reports that this method returns None
            xd = self.eval_discrete_updates(
                context, events
            )  # pylint: disable=assignment-from-none

            # Apply discrete updates
            context = context.with_discrete_state(xd)
            logger.debug(f"Updated discrete state: {context.discrete_state}")
        return context

    def handle_zero_crossings(
        self, events: CompositeEventCollection, context: ContextBase
    ) -> ContextBase:
        """Compute and apply active zero-crossing events.

        This is intended for internal use by the simulator and should not normally need
        to be invoked directly by users. Events are evaluated conditionally on being
        marked "active", so the entire event collection can be passed without filtering
        to active events. This is necessary to make the function calls work with JAX
        tracing, which do not allow for variable-sized arguments or returns.

        Args:
            events (CompositeEventCollection): collection of zero-crossing events
            context (ContextBase): root context for this system

        Returns:
            ContextBase: updated context with all active zero-crossing events applied
        """
        logger.debug(
            "Handling %d state transition events at t=%s",
            events.num_unrestricted_update_events,
            context.time,
        )
        if events.num_unrestricted_update_events > 0:
            state = self.eval_zero_crossing_updates(context, events)
            context = context.with_state(state)
        return context

    def handle_unrestricted_update(
        self, events: CompositeEventCollection, context: ContextBase
    ) -> ContextBase:
        """Compute and apply active unrestricted updates.

        This is intended for internal use by the simulator and should not normally need
        to be invoked directly by users. Events are evaluated conditionally on being
        marked "active", so the entire event collection can be passed without filtering
        to active events. This is necessary to make the function calls work with JAX
        tracing, which do not allow for variable-sized arguments or returns.

        TODO: This is essentially a duplicate of `handle_zero_crossings` but without
        the custom VJP rule. It should be made obsolete with the removal of periodic
        unrestricted updates. See:
        https://collimator.atlassian.net/browse/WC-120

        Args:
            events (CompositeEventCollection): collection of unrestricted update events
            context (ContextBase): root context for this system

        Returns:
            ContextBase: updated context with all active unrestricted updates applied
        """
        logger.debug(
            "Handling %d unrestricted update events at t=%s",
            events.num_unrestricted_update_events,
            context.time,
        )
        if events.num_unrestricted_update_events > 0:
            logger.debug(f"State before eval_unrestricted_updates: {context.state}")
            state = self.eval_unrestricted_updates(context, events)
            logger.debug(f"State after eval_unrestricted_updates: {context.state}")
            context = context.with_state(state)
            logger.debug(f"State after with_state: {context.state}")
        return context

    # Events that happen at some regular interval
    @abc.abstractproperty
    def periodic_events(self) -> CompositeEventCollection:
        pass

    # Events that are triggered by a "guard" function and may induce a "reset" map
    @abc.abstractproperty
    def zero_crossing_events(self) -> CompositeEventCollection:
        pass

    @abc.abstractmethod
    def determine_active_guards(self, context: ContextBase) -> CompositeEventCollection:
        """Determine active guards for zero-crossing events.

        This method is responsible for evaluating and determining which
        zero-crossing events are active based on the current system mode
        and other conditions.  This can be overridden to flag active/inactive
        guards on a block-specific basis, for instance in a StateMachine-type
        block. By default all guards are marked active at this point unless
        the zero-crossing event was declared with a non-default `start_mode`, in
        which case the guard is activated conditionally on the current mode.

        For example, in a system with finite state transitions, where a transition from
        mode A to mode B is triggered by a guard function g_AB and the inverse
        transition is triggered by a guard function g_BA, this function would activate
        g_AB if the system is in mode A and g_BA if the system is in mode B. The other
        guard function would be inactive.  If the zero-crossing event is not associated
        with a starting mode, it is considered to be always active.

        Args:
            root_context (ContextBase):
                The root context containing the overall state and parameters.

        Returns:
            CompositeEventCollection:
                A collection of zero-crossing events with active/inactive status
                updated based on the current system mode and other conditions.
        """
        pass

    #
    # I/O ports
    #
    @property
    def input_ports(self) -> List[InputPort]:
        return [self.cache_sources[i] for i in self.input_port_indices]

    def get_input_port(self, name: str) -> tuple[InputPort, int]:
        """Retrieve a specific input port by name."""
        for i, port in enumerate(self.input_ports):
            if port.name == name:
                return port, i
        raise ValueError(f"System {self.name} has no input port named {name}")

    @property
    def num_input_ports(self) -> int:
        return len(self.input_port_indices)

    @property
    def output_ports(self) -> List[OutputPort]:
        return [self.cache_sources[i] for i in self.output_port_indices]

    def get_output_port(self, name: str) -> OutputPort:
        """Retrieve a specific output port by name."""
        for port in self.output_ports:
            if port.name == name:
                return port
        raise ValueError(f"System {self.name} has no output port named {name}")

    @property
    def num_output_ports(self) -> int:
        return len(self.output_port_indices)

    def eval_input(self, context: ContextBase, port_index: int = 0) -> Array:
        """Get the input for a given port.

        This works by evaluating the callback function associated with the port, which
        will "pull" the upstream output port values.

        Args:
            context (ContextBase): root context for this system
            port_index (int, optional): index into `self.input_ports`, for example
                the value returned by `declare_input_port`. Defaults to 0.

        Returns:
            Array: current input values
        """
        return self.input_ports[port_index].eval(context)

    def collect_inputs(self, context: ContextBase) -> List[Array]:
        """Collect all current inputs for this system.

        Args:
            context (ContextBase): root context for this system

        Returns:
            List[Array]: list of all current input values
        """
        return [self.eval_input(context, i) for i in range(self.num_input_ports)]

    def _eval_input_port(self, context: ContextBase, port_index: int) -> Array:
        """Evaluate an upstream input port given the _root_ context.

        Intended for internal use as a callback function. Users and developers
        should typically call `eval_input` in order to get this information. That
        method will call the callback function associated with the input port,
        which will have a reference to this method.

        Args:
            context (ContextBase): root context for this system
            port_index (int): index of the input port to evaluate on the target system

        Returns:
            Array: current input values
        """
        # A helper function to evaluate an upstream input port given the _root_ context.

        port_locator = self.input_ports[port_index].locator

        if self.parent is None:
            # This is currently the root system.  Typically root input ports should not be evaluated,
            #  but we can get here during subsystem construction (e.g. type inference).  In that case,
            #  we should just defer evaluation and rely on the graph analysis to determine that
            #  everything is connected correctly.
            # See https://collimator.atlassian.net/browse/WC-51.
            # This should not happen during simulation or root context construction.
            logger.debug(
                f"    ---> {self.name} is the root system, deferring evaluation of "
                f"{port_locator[0].name}[{port_locator[1]}]"
            )
            raise UpstreamEvalError(port_locator)

        # The `eval_subsystem_input_port` method is only defined for Diagrams, but
        # the parent system is guaranteed to be a Diagram if this is not the root.
        # If it is the root, it should not have any (un-fixed) input ports.
        return self.parent.eval_subsystem_input_port(context, port_locator)

    #
    # Declaration utilities
    #
    def _next_input_port_name(self, name: str = None) -> str:
        """Automatically generate a unique name for the next input port."""
        if name is not None:
            assert name != ""
            return name
        return f"u_{self.num_input_ports}"

    def _next_output_port_name(self, name: str = None) -> str:
        """Automatically generate a unique name for the next output port."""
        if name is not None:
            assert name != ""
            return name
        return f"y_{self.num_output_ports}"

    def declare_cache(
        self,
        callback: Callable[[ContextBase], Any],
        **kwargs,
    ) -> int:
        """Add a generic cache entry to the system

        This might represent a port, a time derivative, discrete state update, etc.
        Typically one of these should be called directly from the `declare_*` methods.
        Requires a callback function of the form
            `callback(context) -> value`
        which computes the value of the cache given the root context.

        In most cases this should not be needed, but it is provided for cases where
        some arbitrary (non-array) Python object needs to be stored in the context.
        When this is used, the `clear_cache` method should likely also be
        overridden to ensure that whatever is stored is not deleted.

        TODO: At this moment this is only used for creating the ODE RHS function and
        in the ModelicaFMU block.  In the latter case, it is used to store the FMU
        in the cache, but in reality that system actually still just stores the FMU
        as a block attribute.  Really, this code should probably just be moved to
        `declare_continuous_state` and the ModelicaFMU should be updated to reflect
        what it actually does.  On the other hand, if the FMU can be made to work
        with the cache, then it would prove that there are cases where this is still
        useful.

        Args:
            callback (Callable): computes the value of the cache given root context
            **kwargs: additional keyword arguments to pass to the CacheSource
                constructor.  For example, `name` or `prerequisites_of_calc`.

        Returns:
            int:
                index of the newly created CacheSource in the system cache_sources list
        """
        cache = CacheSource(
            callback,
            system=self,
            cache_index=len(self.cache_sources),
            **kwargs,
        )
        self.cache_sources.append(cache)
        return cache.cache_index

    def declare_input_port(
        self,
        name: str = None,
        prerequisites_of_calc: List[DependencyTicket] = None,
    ) -> int:
        """Add an input port to the system.

        Returns the corresponding index into the system input_port_indices list
        Note that this is different from the cache_sources index - typically it
        will make more sense to retrieve via system.input_ports[port_index], but

        Args:
            name (str, optional): name of the new port. Defaults to None, which will
                use the default naming scheme for the system (e.g. "u_0")
            prerequisites_of_calc (List[DependencyTicket], optional): list of
                dependencies for the callback function. Defaults to None.

        Returns:
            int: port index of the newly created port in `input_ports`
        """
        port_index = self.num_input_ports
        port_name = self._next_input_port_name(name)

        for port in self.input_ports:
            assert (
                port.name != port_name
            ), f"System {self.name} already has an input port named {port.name}"

        def _callback(context: ContextBase) -> Array:
            # Given the root context, evaluate the input port using the helper function
            return self._eval_input_port(context, port_index)

        cache_index = len(self.cache_sources)
        port = InputPort(
            _callback,
            system=self,
            cache_index=cache_index,
            name=port_name,
            index=port_index,
            prerequisites_of_calc=prerequisites_of_calc,
        )

        assert isinstance(port, InputPort)
        assert port.system is self
        assert port.name != ""

        # Check that name is unique
        for p in self.input_ports:
            assert (
                p.name != port.name
            ), f"System {self.name} already has an input port named {port.name}"

        self.input_port_indices.append(cache_index)
        self.cache_sources.append(port)

        return port_index

    def declare_output_port(
        self,
        callback: Callable,
        name: str = None,
        prerequisites_of_calc: List[DependencyTicket] = None,
        default_value: Array = None,
    ) -> int:
        """Add an output port to the system.

        This output port could represent any function of the context available to
        the system, so a callback function is required.  This function should have
        the form
            `callback(context: ContextBase) -> Array`
        SystemBase implementations have some specific convenience wrappers, e.g.:
            `LeafSystem.declare_continuous_state_output_port`
            `LeafSystem.declare_discrete_state_output_port`
            `LeafSystem.export_output`

        Common cases are:
        - Feedthrough blocks: gather inputs and return some function of the
            inputs (e.g. a gain)
        - Stateful blocks: use LeafSystem.declare_(...)_state_output_port to
            return the value of a particular state
        - Diagrams: create and export a diagram-level port to the parent system using
            the callback function associated with the system-level port

        Returns the corresponding index into the system output_port_indices list
        Note that this is different from the cache_sources index - typically it
        will make more sense to retrieve via system.output_ports[port_index], but

        Args:
            callback (Callable): computes the value of the output port given
                the root context.
            name (str, optional): name of the new port. Defaults to None, which will
                use the default naming scheme for the system (e.g. "y_0")
            prerequisites_of_calc (List[DependencyTicket], optional): list of
                dependencies for the callback function. Defaults to None, which will
                use the default dependencies for the system (all sources).  This may
                conservatively flag the system as having algebraic loops, so it is
                better to be specific here when possible.  This is done automatically
                in the wrapper functions like `LeafSystem.declare_(...)_output_port`
            default_value (Array, optional): A default array-like value used to seed
                the context and perform type inference, when this is known up front.
                Defaults to None, which will use information propagation through the
                graph along with type promotion to determine an appropriate value.

        Returns:
            int: port index of the newly created port
        """
        port_index = self.num_output_ports
        port_name = self._next_output_port_name(name)

        for port in self.output_ports:
            assert (
                port.name != port_name
            ), f"System {self.name} already has an output port named {port.name}"

        if prerequisites_of_calc is None:
            prerequisites_of_calc = [DependencyTicket.all_sources]

        cache_index = len(self.cache_sources)
        port = OutputPort(
            callback,
            system=self,
            cache_index=cache_index,
            name=port_name,
            index=port_index,
            prerequisites_of_calc=prerequisites_of_calc,
            default_value=default_value,
        )

        assert isinstance(port, OutputPort)
        assert port.system is self
        assert port.name != ""

        # Check that name is unique
        for p in self.output_ports:
            assert (
                p.name != port.name
            ), f"System {self.name} already has an output port named {port.name}"

        logger.debug(f"Adding output port {port} to {self.name}")
        self.output_port_indices.append(cache_index)
        self.cache_sources.append(port)
        logger.debug(
            f"    ---> {self.name} now has {len(self.output_ports)} output ports: {self.output_ports}"
        )
        logger.debug(
            f"    ---> {self.name} now has {len(self.cache_sources)} cache sources: {self.cache_sources}"
        )

        return port_index

    @abc.abstractmethod
    def get_feedthrough(self) -> List[Tuple[int, int]]:
        """Determine pairs of direct feedthrough ports for this system.

        By default, the algorithm relies on the dependency tracking system to determine
        feedthrough, but this can be overridden by implementing this method directly
        in a subclass, for instance if the automatic dependency tracking is too
        conservative in determining feedthrough.

        Returns:
            List[Tuple[int, int]]:
                A list of tuples (u, v) indicating that input port u is a direct
                feedthrough to output port v.  The indices u and v correspond to the
                indices of the input and output ports in the system's input and output
                port lists.
        """
        pass

    #
    # Initialization
    #
    def create_context(self, **kwargs) -> ContextBase:
        """Create a new context for this system.

        The context will contain all variable information used in
        simulation/analysis/optimization, such as state and parameters.

        Returns:
            ContextBase: new context for this system
        """
        return self.context_factory(**kwargs)

    def clear_cache(self, cache: Cache) -> Cache:
        """Clear unused data from the cache to avoid carrying through compilation."""
        return {}

    def check_types(self, context: ContextBase):
        """Perform any system-specific static analysis."""
        pass

    @abc.abstractproperty
    def context_factory(self) -> ContextFactory:
        """Factory object for creating contexts for this system.

        Should not be called directly - use `system.create_context` instead.
        """
        pass

    @property
    def dependency_graph(self) -> DependencyGraph:
        """Retrieve (or create if necessary) the dependency graph for this system."""
        if self._dependency_graph is None:
            self._dependency_graph = self.create_dependency_graph()
        return self._dependency_graph

    @abc.abstractproperty
    def dependency_graph_factory(self) -> DependencyGraphFactory:
        """Factory object for creating dependency graphs for this system.

        Should not be called directly - use `system.create_dependency_graph` instead.
        """
        pass

    def create_dependency_graph(self) -> DependencyGraph:
        """Create a dependency graph for this system."""
        return self.dependency_graph_factory()

    def initialize_static_data(self, context: ContextBase) -> ContextBase:
        """Initialize any context data that has to be done after context creation.

        Use this to define custom auxiliary data or type inference that doesn't
        get traced by JAX. See the `ZeroOrderHold` implementation for an example.
        Since this is only applied during context initialization, it is allowed to
        modify the context directly (or the system itself).

        Typically this should not be called outside of the ContextFactory.

        Args:
            context (ContextBase): partially initialized context for this system.
        """
        return context

    @property
    def ports(self) -> dict[str, PortBase]:
        """Dictionary of all ports in this system, indexed by name"""
        return {port.name: port for port in self.input_ports + self.output_ports}
