"""Event classes for hybrid system simulation.

This module defines classes used for event-driven simulation of hybrid systems.
These classes are used internally by the simulation framework and should not
normally need to be used directly by users. Instead, users can declare events
on LeafSystems.  The events will be organized into CompositeEventCollections and
handled by the simulation framework.
"""
from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Any, Callable, List, Hashable
from collections import OrderedDict

from jax import tree_util
import jax.numpy as jnp

from ..math_backend import inf, nan, cond


if TYPE_CHECKING:
    from ..math_backend.typing import Scalar, Array
    from .context import ContextBase
    from .state import LeafState

__all__ = [
    "DiscreteUpdateEvent",
    "PublishEvent",
    "UnrestrictedUpdateEvent",
    "ZeroCrossingEvent",
    "PeriodicEventData",
    "ZeroCrossingEventData",
    "CompositeEventCollection",
    "LeafCompositeEventCollection",
    "DiagramCompositeEventCollection",
    "is_event_data",
]


@dataclasses.dataclass(frozen=True)
class EventData:
    active: bool


@dataclasses.dataclass(frozen=True)
class PeriodicEventData(EventData):
    period: float  # Period of the event
    offset: float  # Offset from the start of the simulation for the initial event

    # Time of the next event sample, as determined by the simulation loop.
    next_sample_time: float = inf


@dataclasses.dataclass(frozen=True)
class ZeroCrossingEventData(EventData):
    w0: Scalar = inf  # Guard value at beginning of interval
    w1: Scalar = inf  # Guard value at end of interval


#
# Trigger functions for zero-crossing events
#
def _none_trigger(w0: Scalar, w1: Scalar) -> bool:
    return False


def _positive_then_nonpositive_trigger(w0: Scalar, w1: Scalar) -> bool:
    return (w0 > 0) & (w1 <= 0)


def _negative_then_nonnegative_trigger(w0: Scalar, w1: Scalar) -> bool:
    return (w0 < 0) & (w1 >= 0)


def _crosses_zero_trigger(w0: Scalar, w1: Scalar) -> bool:
    return ((w0 > 0) & (w1 <= 0)) | ((w0 < 0) & (w1 >= 0))


def _edge_detection(w0: Scalar, w1: Scalar) -> bool:
    return w0 != w1


_zero_crossing_trigger_functions = {
    "none": _none_trigger,
    "positive_then_non_positive": _positive_then_nonpositive_trigger,
    "negative_then_non_negative": _negative_then_nonnegative_trigger,
    "crosses_zero": _crosses_zero_trigger,
    "edge_detection": _edge_detection,
}


#
# Event classes
#
def is_event_data(x: Any) -> bool:
    return isinstance(x, EventData)


def _activate(self, activation_fn):
    """Map a bool-valued activation function over all events in the tree."""

    def _activate_helper(event_data):
        if is_event_data(event_data):
            return dataclasses.replace(event_data, active=activation_fn(event_data))
        return event_data

    return tree_util.tree_map(
        _activate_helper,
        self,
        is_leaf=is_event_data,
    )


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class Event:
    """Class representing a discontinuous update event in a hybrid system.

    Users should not need to interact with these objects directly. They are intended
    to be used internally by the simulation framework for handling events in hybrid
    system simulation. In a normal workflow, events will be declared on LeafSystems
    using `declare_*` methods, and the simulation framework will organize them into
    CompositeEventCollections.
    """

    event_data: EventData = None

    # The callback function is called when the event is triggered. The callback will
    # be passed the root context, but the return value will vary depending on the event
    # type (as defined by subclass implementations).
    callback: Callable[[ContextBase], Any] = None

    # The "passthrough" is a dummy callback that happens if the real callback
    # is not active (False branch of conditional). This is required to have a
    # consistent signature with `callback` for `lax.cond`.
    passthrough: Callable[[ContextBase], Any] = None

    # The state index is used to specify a particular discrete state component which
    # is updated by the event. This is only used for discrete update events.
    state_index: int = None

    # If true, the update calls will use the structured control flow provided by LAX
    # rather than the standard Python control flow.
    enable_tracing: bool = True

    def __post_init__(self):
        if self.passthrough is None:

            def _default_passthrough(context: ContextBase) -> ContextBase:
                return context

            self.passthrough = _default_passthrough

    def __repr__(self) -> str:
        return f"{self.event_data}"

    # Proper typing here is difficult because the return type of the callback will
    # vary depending on the type of callback (determined by the subclass). For example,
    # the callback in a DiscreteUpdateEvent will return an Array, while unrestricted
    # updates return a LeafState.
    def handle(self, context: ContextBase) -> Any:
        """Conditionally compute the result of the update callback

        If the event is marked "inactive" via its event data attribute, the passthrough
        callback will be called instead of the update callback. Otherwise, the update
        callback will be called. The return types of both callbacks must match, but the
        specific type will depend on the kind of event.
        """
        if self.enable_tracing:
            return cond(
                self.event_data.active,
                self.callback,
                self.passthrough,
                context,
            )

        # No tracing: use standard control flow
        if not self.event_data.active:
            return self.passthrough(context)
        return self.callback(context)

    def mark_active(self) -> Event:
        """Create a copy of the event with the status marked active"""
        return _activate(self, lambda _: True)

    def mark_inactive(self) -> Event:
        """Create a copy of the event with the status marked inactive"""
        return _activate(self, lambda _: False)

    #
    # PyTree registration
    #
    # Normally it's convenient to move this out of the class definition, but because
    # there are several subclasses that can all be registered in the same way, having
    # it defined here allows code reuse.
    def tree_flatten(self):
        children = (self.event_data,)
        aux_data = (
            self.callback,
            self.passthrough,
            self.state_index,
            self.enable_tracing,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (event_data,) = children
        callback, passthrough, state_index, enable_tracing = aux_data
        return cls(
            event_data=event_data,
            callback=callback,
            passthrough=passthrough,
            state_index=state_index,
            enable_tracing=enable_tracing,
        )


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class DiscreteUpdateEvent(Event):
    """Event representing a discrete update in a hybrid system."""

    # Supersede type hints in Event with the specific signature for discrete updates
    callback: Callable[[ContextBase], Array] = None
    passthrough: Callable[[ContextBase], Array] = None

    # Inherits docstring from Event. This is only needed to specialize type hints.
    def handle(self, context: ContextBase) -> Array:
        return super().handle(context)


# TODO: deprecate publish events
@tree_util.register_pytree_node_class
@dataclasses.dataclass
class PublishEvent(Event):
    pass


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class UnrestrictedUpdateEvent(Event):
    """Event representing an update to any state components in a hybrid system."""

    # Supersede type hints in Event with the specific signature for unrestricted updates
    callback: Callable[[ContextBase], LeafState] = None
    passthrough: Callable[[ContextBase], LeafState] = None

    # Inherits docstring from Event. This is only needed to specialize type hints.
    def handle(self, context: ContextBase) -> LeafState:
        return super().handle(context)


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class ZeroCrossingEvent(UnrestrictedUpdateEvent):
    """An event that triggers when a specified "guard" function crosses zero.

    The event is triggered when the guard function crosses zero in the specified
    direction. In addition to the guard callback, the event also has a "reset map"
    which is called when the event is triggered. The reset map may update any state
    component in the system.

    The "direction" of the zero-crossing is one of the following:
        - "none": Never trigger the event (can be useful for debugging)
        - "positive_then_non_positive": Trigger when the guard goes from positive to
            non-positive
        - "negative_then_non_negative": Trigger when the guard goes from negative to
            non-negative
        - "crosses_zero": Trigger when the guard crosses zero in either direction
        - "edge_detection": Trigger when the guard changes value

    Notes:
        This class should typically not need to be used directly by users. Instead,
        declare the guard function and reset map on a LeafSystem using the
        `declare_zero_crossing` method.  The event will then be auto-generated for
        simulation.
    """

    guard: Callable[[ContextBase], Scalar] = None
    reset_map: dataclasses.InitVar[Callable[[ContextBase], LeafState]] = None
    direction: str = "crosses_zero"
    name: str = None
    event_data: ZeroCrossingEventData = None

    # If not none, only trigger when in this mode. This logic is handled by the owning
    # leaf system.
    active_mode: int = None

    def __post_init__(self, reset_map):
        self.callback = reset_map

    def _should_trigger(self, w0: Scalar, w1: Scalar) -> bool:
        """Determine if the event should trigger.

        This will use the provided beginning/ending guard value (w0 and w1, resp.),
        as well as the direction of the zero-crossing event. Additionally, the event
        will only trigger if it has been marked as "active", indicating for example
        that the system is in the correct "mode" or "stage" from which the event might
        trigger.
        """
        active = self.event_data.active

        trigger_func = _zero_crossing_trigger_functions[self.direction]
        return active & trigger_func(w0, w1)

    def should_trigger(self) -> bool:
        """Determine if the event should trigger based on the stored guard values."""
        return self._should_trigger(self.event_data.w0, self.event_data.w1)

    def eval_guard(self, context: ContextBase) -> Scalar:
        """Evaluate the guard function given the _root_ context."""
        if self.enable_tracing:
            g = self.guard(context)
            return jnp.where(self.event_data.active, g, nan)

        # No tracing: use standard control flow
        if self.event_data.active:
            return self.guard(context)
        return nan

    #
    # PyTree registration
    #
    def tree_flatten(self):
        children = (self.event_data,)
        aux_data = (
            self.guard,
            self.callback,
            self.name,
            self.direction,
            self.passthrough,
            self.enable_tracing,
            self.active_mode,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            event_data=children[0],
            guard=aux_data[0],
            reset_map=aux_data[1],
            name=aux_data[2],
            direction=aux_data[3],
            passthrough=aux_data[4],
            enable_tracing=aux_data[5],
            active_mode=aux_data[6],
        )


#
# Event collections
#
class CompositeEventCollection(metaclass=abc.ABCMeta):
    """A collection of events owned by a system.

    Users should not need to interact with these objects directly. They are intended
    to be used internally by the simulation framework for handling events in hybrid
    system simulation.

    These contain callback functions that update the context in various ways
    when the event is triggered. There will be different "collections" for each
    trigger type in simulation (e.g. periodic vs zero-crossing). Within the
    collections, events are broken out by function (e.g. discrete vs unrestricted
    updates).

    There are separate implementations for leaf and diagram systems, where the
    DiagramCompositeEventCollection preserves the tree structure of the underlying
    Diagram. However, the interface in both cases is the same and is identical to
    the interface defined by CompositeEventCollection.
    """

    @abc.abstractmethod
    def __getitem__(self, key: Hashable) -> CompositeEventCollection:
        pass

    @abc.abstractproperty
    def publish_events(self) -> List[PublishEvent]:
        pass

    @abc.abstractproperty
    def num_publish_events(self) -> int:
        pass

    @abc.abstractproperty
    def discrete_update_events(self) -> List[DiscreteUpdateEvent]:
        pass

    @abc.abstractproperty
    def num_discrete_update_events(self) -> int:
        pass

    @abc.abstractproperty
    def unrestricted_update_events(self) -> List[UnrestrictedUpdateEvent]:
        pass

    @abc.abstractproperty
    def num_unrestricted_update_events(self) -> int:
        pass

    @abc.abstractmethod
    def activate(self, activation_fn) -> CompositeEventCollection:
        pass

    def mark_all_active(self) -> CompositeEventCollection:
        return self.activate(lambda _: True)

    def mark_all_inactive(self) -> CompositeEventCollection:
        return self.activate(lambda _: False)

    @property
    def num_active(self) -> int:
        def _get_active(event_data: EventData) -> bool:
            return event_data.active

        active_tree = tree_util.tree_map(
            _get_active,
            self,
            is_leaf=is_event_data,
        )
        return sum(tree_util.tree_leaves(active_tree))

    @property
    def has_active(self) -> bool:
        return self.num_active > 0

    @abc.abstractproperty
    def has_events(self) -> bool:
        pass

    def pprint(self, output=print):
        output(self._pprint_helper().strip())

    def _pprint_helper(self, prefix="") -> str:
        s = f"{prefix}|-- \n"
        if len(self.publish_events) > 0:
            s += f"{prefix}    Publish Events:\n"
            for event in self.publish_events:
                s += f"{prefix}    |  {event}\n"
        if len(self.discrete_update_events) > 0:
            s += f"{prefix}    Discrete Update Events:\n"
            for event in self.discrete_update_events:
                s += f"{prefix}    |  {event}\n"
        if len(self.unrestricted_update_events) > 0:
            s += f"{prefix}    Unrestricted Update Events:\n"
            for event in self.unrestricted_update_events:
                s += f"{prefix}    |  {event}\n"
        return s

    def __repr__(self) -> str:
        s = f"{type(self).__name__}("
        if self.num_discrete_update_events > 0:
            s += f"discrete_update: {self.discrete_update_events} "
        if self.num_publish_events > 0:
            s += f"publish: {self.publish_events} "
        if self.num_unrestricted_update_events > 0:
            s += f"unrestricted_update: {self.unrestricted_update_events}"
        s += ")"

        return s


# This will be registered as a pytree for use in simulation - it should be treated
#  as mutable during system construction, but immutable once the system is finalized.
@dataclasses.dataclass
class LeafCompositeEventCollection(CompositeEventCollection):
    _discrete_update_events: List[DiscreteUpdateEvent] = dataclasses.field(
        default_factory=list
    )
    _publish_events: List[PublishEvent] = dataclasses.field(default_factory=list)
    _unrestricted_update_events: List[UnrestrictedUpdateEvent] = dataclasses.field(
        default_factory=list
    )

    def __getitem__(self, key: Hashable) -> LeafCompositeEventCollection:
        return self

    @property
    def discrete_update_events(self) -> List[DiscreteUpdateEvent]:
        return self._discrete_update_events

    @property
    def num_discrete_update_events(self) -> int:
        return len(self.discrete_update_events)

    @property
    def publish_events(self) -> List[PublishEvent]:
        return self._publish_events

    @property
    def num_publish_events(self) -> int:
        return len(self.publish_events)

    @property
    def unrestricted_update_events(self) -> List[UnrestrictedUpdateEvent]:
        return self._unrestricted_update_events

    @property
    def num_unrestricted_update_events(self) -> int:
        return len(self.unrestricted_update_events)

    def add_discrete_update_event(self, event: DiscreteUpdateEvent):
        self.discrete_update_events.append(event)

    def add_publish_event(self, event: PublishEvent):
        self.publish_events.append(event)

    def add_unrestricted_update_event(self, event: UnrestrictedUpdateEvent):
        self.unrestricted_update_events.append(event)

    def __add__(
        self, other: LeafCompositeEventCollection
    ) -> LeafCompositeEventCollection:
        return LeafCompositeEventCollection(
            _discrete_update_events=self.discrete_update_events
            + other.discrete_update_events,
            _publish_events=self.publish_events + other.publish_events,
            _unrestricted_update_events=self.unrestricted_update_events
            + other.unrestricted_update_events,
        )

    @property
    def has_events(self) -> bool:
        return (
            len(self.publish_events) > 0
            or len(self.discrete_update_events) > 0
            or len(self.unrestricted_update_events) > 0
        )

    def activate(self, activation_fn) -> LeafCompositeEventCollection:
        return _activate(self, activation_fn)


@dataclasses.dataclass(frozen=True)
class DiagramCompositeEventCollection(CompositeEventCollection):
    subevent_collection: OrderedDict[
        Hashable, LeafCompositeEventCollection
    ] = dataclasses.field(default_factory=OrderedDict)

    def __getitem__(self, key: Hashable) -> LeafCompositeEventCollection:
        return self.subevent_collection[key]

    @property
    def num_subevents(self) -> int:
        return len(self.subevent_collection)

    @property
    def num_publish_events(self) -> int:
        return sum(
            [
                subevent.num_publish_events
                for subevent in self.subevent_collection.values()
            ]
        )

    @property
    def publish_events(self) -> List[PublishEvent]:
        # Return a flattened list of publish events
        publish_events = []
        for subevent in self.subevent_collection.values():
            publish_events.extend(subevent.publish_events)
        return publish_events

    @property
    def num_discrete_update_events(self) -> int:
        return sum(
            [
                subevent.num_discrete_update_events
                for subevent in self.subevent_collection.values()
            ]
        )

    @property
    def discrete_update_events(self) -> List[DiscreteUpdateEvent]:
        # Return a flattened list of discrete update events
        discrete_update_events = []
        for subevent in self.subevent_collection.values():
            discrete_update_events.extend(subevent.discrete_update_events)
        return discrete_update_events

    @property
    def num_unrestricted_update_events(self) -> int:
        return sum(
            [
                subevent.num_unrestricted_update_events
                for subevent in self.subevent_collection.values()
            ]
        )

    @property
    def unrestricted_update_events(self) -> List[UnrestrictedUpdateEvent]:
        # Return a flattened list of unrestricted update events
        unrestricted_update_events = []
        for subevent in self.subevent_collection.values():
            unrestricted_update_events.extend(subevent.unrestricted_update_events)
        return unrestricted_update_events

    def __add__(
        self, other: DiagramCompositeEventCollection
    ) -> DiagramCompositeEventCollection:
        assert self.num_subevents == other.num_subevents
        subevent_collection = OrderedDict()
        for sys_id in self.subevent_collection:
            subevent_collection[sys_id] = (
                self.subevent_collection[sys_id] + other.subevent_collection[sys_id]
            )
        return DiagramCompositeEventCollection(subevent_collection)

    @property
    def has_events(self) -> bool:
        for subevents in self.subevent_collection.values():
            if subevents.has_events:
                return True
        return False

    def activate(self, activation_fn) -> DiagramCompositeEventCollection:
        subevent_collection = OrderedDict(
            {
                sys_id: subevents.activate(activation_fn)
                for sys_id, subevents in self.subevent_collection.items()
            }
        )
        return dataclasses.replace(self, subevent_collection=subevent_collection)


#
# PyTree registration
#
def periodic_event_data_flatten(event_data: PeriodicEventData):
    children = (
        event_data.active,
        event_data.next_sample_time,
        event_data.period,
        event_data.offset,
    )
    aux_data = ()
    return children, aux_data


def periodic_event_data_unflatten(aux_data, children):
    active, next_sample_time, period, offset = children
    return PeriodicEventData(
        active=active,
        period=period,
        offset=offset,
        next_sample_time=next_sample_time,
    )


tree_util.register_pytree_node(
    PeriodicEventData,
    periodic_event_data_flatten,
    periodic_event_data_unflatten,
)


def zero_crossing_data_flatten(event_data: ZeroCrossingEventData):
    children = (event_data.active, event_data.w0, event_data.w1)
    aux_data = ()
    return children, aux_data


def zero_crossing_data_unflatten(aux_data, children):
    active, w0, w1 = children
    return ZeroCrossingEventData(
        active=active,
        w0=w0,
        w1=w1,
    )


tree_util.register_pytree_node(
    ZeroCrossingEventData,
    zero_crossing_data_flatten,
    zero_crossing_data_unflatten,
)


def leaf_collection_flatten(collection: LeafCompositeEventCollection):
    children = (
        collection.discrete_update_events,
        collection.publish_events,
        collection.unrestricted_update_events,
    )
    aux_data = ()
    return children, aux_data


def leaf_collection_unflatten(aux_data, children):
    return LeafCompositeEventCollection(
        _discrete_update_events=children[0],
        _publish_events=children[1],
        _unrestricted_update_events=children[2],
    )


tree_util.register_pytree_node(
    LeafCompositeEventCollection,
    leaf_collection_flatten,
    leaf_collection_unflatten,
)


def diagram_collection_flatten(collection: DiagramCompositeEventCollection):
    children = collection.subevent_collection.values()
    aux_data = collection.subevent_collection.keys()
    return children, aux_data


def diagram_collection_unflatten(aux_data, children):
    subevent_collection = OrderedDict(
        {key: val for key, val in zip(aux_data, children)}
    )
    return DiagramCompositeEventCollection(subevent_collection)


tree_util.register_pytree_node(
    DiagramCompositeEventCollection,
    diagram_collection_flatten,
    diagram_collection_unflatten,
)
