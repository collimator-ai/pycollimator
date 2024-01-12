"""Functionality for tracking dependencies in a system.

Nothing in this file should normally need to be directly accessed by users.
A dependency graph will be created automatically by an owning system as necessary,
using the DependencyGraphFactory classes.

This dependency system was originally based on the Drake caching system:

https://drake.mit.edu/doxygen_cxx/group__cache__design__notes.html

However, a direct port of that caching system turned out to be extremely inefficient
in JAX.  Due to the need to trace through all the control flow involved in detailed
cache invalidation, compile times suffered noticeably without any major improvement
in runtime for "typical" systems.

As a result, this system is currently barely used (only for determining LeafSystem
feedthrough during automatic loop detection), but is preserved in case it is useful
for when function ordering replaces the current lazy evaluation model.

The basic idea is that any value on which a computation might depend is associated
with an integer "dependency ticket" managed by the DependencyTicket singleton class.
These tickets may be associated with a specific system-level value (e.g. a parameter
or port value), or they may be composites that represent a group of tickets (e.g.
`DependencyTicket.x` for all state variables.

On a system level, all "cache sources" (usually these are ports or ODE RHS functions),
discrete state components, and parameters are each associated with a dependency ticket.
A "dependency graph" is a system-level mapping from these dependency tickets to
DependencyTracker objects, which can contain references to "prerequisites" (other
trackers on which the current tracker depends) and "subscribers" (other
trackers that depend on the current tracker).  Diagram-level trackers may
therefore reference leaf system trackers as prerequisites or subscribers via composite
tickets like "all input ports" (`DependencyTicket.u`).

If a dependency tracker is associated with a callback function, it also contains enough
information to uniquely identify that function given the corresponding system.

The resulting dependency graph contains a complete picture of information flow in the
system.  The DAG formed by prerequisites of a tracker will include all calculations
required to compute the value associated that tracker.  For instance, given the
dependency ticket for a particular port, `dependency_graph[port_ticket]` will be a
tracker with the direct prerequisites of the port value calculation.  Those
prerequisite trackers will in turn have their own prerequisites, and so on, until the
information propagation reaches "known" values like state variables or parameters.

Currently all this information is unused, but it could for example be used to construct
an explicit graph of function calls for a given system, which could then be sorted
into a more efficient order for compilation.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, List, Mapping, Hashable

if TYPE_CHECKING:
    from .cache import Cache
    from .system_base import SystemBase
    from .leaf_system import LeafSystem
    from .diagram import Diagram

__all__ = [
    "DependencyTicket",
    "next_dependency_ticket",
    "mark_cache",
    "DependencyGraph",
    "LeafDependencyGraphFactory",
    "DiagramDependencyGraphFactory",
]


def next_dependency_ticket():
    """Create a new unique dependency ticket using the next available value."""
    return DependencyTicket.next_available_ticket()


class DependencyTicket:
    """Singleton class for managing unique dependency tickets."""

    nothing = 0  # Indicates "not dependent on anything".
    time = 1  # Time.
    xc = 2  # All continuous state variables.
    xd = 3  # All discrete state variables
    mode = 4  # All modes.
    x = 5  # All state variables x = {xc, xd, mode}.
    p = 6  # All parameters
    all_sources_except_input_ports = 7  # Everything except input ports.
    u = 8  # All input ports u.
    all_sources = 9  # All of the above.
    xcdot = 10  # Continuous state time derivative

    _next_available = 11  # This will get incremented by next_available_ticket().

    @classmethod
    def next_available_ticket(cls):
        cls._next_available += 1
        return cls._next_available


def mark_cache(cache: Cache, cache_index: int, is_out_of_date: bool):
    new_cache_entry = cache[cache_index]._replace(is_out_of_date=is_out_of_date)
    return {**cache, cache_index: new_cache_entry}


# This is static data so shouldn't need to be registered as a pytree.
#   It DOES need to be static once created, though.
class DependencyTracker:
    def __init__(
        self,
        description: str,
        ticket: DependencyTicket,
        system_id: Hashable = None,
        cache_index: int = None,
    ):
        self.description = description
        self.ticket = ticket

        # System ID of the owning system.  This is not needed besides for debugging.
        self.system_id = system_id

        self.cache_index = cache_index
        self._prerequisites: List[DependencyTracker] = []
        self._subscribers: List[DependencyTracker] = []

        self._is_finalized = False  # Set this when the context is created

    def add_prerequisite(self, prerequisite: DependencyTracker):
        # This was SubscribeToPrerequisite in v1/Drake
        assert not self._is_finalized, "Cannot modify a finalized dependency tracker"
        # logger.debug(
        #     f"    Adding prerequisite {prerequisite.description} to {self.description}"
        # )
        self._prerequisites.append(prerequisite)
        prerequisite.add_subscriber(self)

    def add_subscriber(self, subscriber: DependencyTracker):
        # This was AddSubscriber in v1/Drake
        assert not self._is_finalized, "Cannot modify a finalized dependency tracker"
        # logger.debug(
        #     f"    Adding subscriber {subscriber.description} to {self.description}"
        # )
        self._subscribers.append(subscriber)

    def notify_subscribers(
        self, cache: Cache, dependency_graph: DependencyGraph
    ) -> Cache:
        for subscriber in self._subscribers:
            # logger.debug(
            #     f"    Notifying {subscriber.description} of change to {self.description}"
            # )
            cache = subscriber.note_prerequisite_change(cache, dependency_graph)
        return cache

    def note_prerequisite_change(
        self, cache: Cache, dependency_graph: DependencyGraph
    ) -> Cache:
        if self.cache_index is not None:
            # logger.debug(f"    Invalidating cache {self.cache_index} for {self.ticket}")
            cache = mark_cache(cache, self.cache_index, is_out_of_date=True)
        return self.notify_subscribers(cache, dependency_graph=dependency_graph)

    def __repr__(self) -> str:
        return f"DependencyTracker({self.description})"

    def finalize(self):
        self._is_finalized = True


def _generic_dependency_graph(
    system_id: Hashable,
) -> Mapping[DependencyTicket, DependencyTracker]:
    # This function was: ContextBase::CreateBuiltInTrackers in Drake

    generic_ticket_names = (
        "nothing",
        "time",
        "xc",
        "xd",
        "mode",
        "x",
        "p",
        "all_sources_except_input_ports",
        "u",
        "all_sources",
        "xcdot",
    )
    generic_tickets = map(
        lambda name: getattr(DependencyTicket, name), generic_ticket_names
    )
    graph = {
        ticket: DependencyTracker(name, ticket, system_id=system_id)
        for (name, ticket) in zip(generic_ticket_names, generic_tickets)
    }  # DependencyTicket -> DependencyTracker

    # Complete state depends on continuous, discrete, and abstract states
    graph[DependencyTicket.x].add_prerequisite(graph[DependencyTicket.xc])
    graph[DependencyTicket.x].add_prerequisite(graph[DependencyTicket.xd])
    graph[DependencyTicket.x].add_prerequisite(graph[DependencyTicket.mode])

    # "All input ports" u tracker.  The associated System is responsible for allocating the
    # individual input port uᵢ trackers and subscribing to them.

    ticket = DependencyTicket.all_sources_except_input_ports
    graph[ticket].add_prerequisite(graph[DependencyTicket.time])
    graph[ticket].add_prerequisite(graph[DependencyTicket.x])
    graph[ticket].add_prerequisite(graph[DependencyTicket.p])

    # "All sources" tracker.
    ticket = DependencyTicket.all_sources
    graph[ticket].add_prerequisite(
        graph[DependencyTicket.all_sources_except_input_ports]
    )
    graph[ticket].add_prerequisite(graph[DependencyTicket.u])

    return graph


def print_dependency_graph(graph: Mapping[DependencyTicket, DependencyTracker]) -> str:
    def _repr_helper(tracker: DependencyTracker, indent: int) -> str:
        repr = ""
        repr += f"{'  ' * indent}{tracker.description}\n"
        repr += f"{'  ' * indent}  Subscribers: {[s.description for s in tracker._subscribers]}\n"
        repr += f"{'  ' * indent}  Prerequisites: {[p.description for p in tracker._prerequisites]}\n"
        for sub in tracker._prerequisites:
            repr += _repr_helper(sub, indent + 1)
        return repr

    repr = ""
    for ticket, tracker in graph.items():
        repr += _repr_helper(tracker, 1)
    return repr


#
# Type hints
#
DependencyGraph = Mapping[DependencyTicket, DependencyTracker]


class DependencyGraphFactory(metaclass=abc.ABCMeta):
    """Factory class for creating the dependency graph for a specific system.

    Since the dependency graph is not really a core part of the system interface
    and is currently only used for static pre-processing (algebraic loop detection),
    creation is outsourced to a factory class to streamline the core System code.

    This should not normally need to be accessed directly - the factory class and
    dependency graph will be created automatically by the owning system as necessary.
    """

    def __init__(self, system: SystemBase):
        self.system = system

    def __call__(self) -> DependencyGraph:
        """Create a dependency graph for the owning system.

        The dependency graph has structure dict[DependencyTicket, DependencyTracker].
        The tickets and trackers will be specific to the owning system, but the
        trackers may have prerequisites or subscribers referencing trackers belonging
        to other systems.
        """

        system = self.system

        # Create trackers for "generic" tickets like "time", "all_sources", etc.
        self.dependency_graph = _generic_dependency_graph(system.system_id)

        # Subscribe to system-specific sources
        for source in system.cache_sources:
            self.dependency_graph[source.ticket] = DependencyTracker(
                source.name,
                source.ticket,
                system_id=system.system_id,
                cache_index=source.cache_index,
            )

        # Subscribe "composite" trackers like xc, xcdot, etc.
        self.subscribe_composite_trackers()

        # The composite "all inputs" u tracker should now be subscribed to all input ports
        for port in system.input_ports:
            if port.is_fixed:
                continue
            tracker = self.dependency_graph[DependencyTicket.u]
            tracker.add_prerequisite(self.dependency_graph[port.ticket])

        # For diagrams, subscribe internal and exported ports to each other
        # For leaf systems, this won't do anything
        self.subscribe_ports()

        return self.dependency_graph

    @abc.abstractmethod
    def subscribe_composite_trackers(self):
        """Hook for subscribing to composite trackers in graph creation.

        Composite trackers represent groups of other trackers, for example
        "all continuous state variables" or "all input ports".
        """
        pass

    def subscribe_ports(self):
        """Hook for subscribing to ports in graph creation.

        The default implementation (for leaf nodes) is to do nothing.
        For diagrams, this will subscribe internal and exported ports to each other.
        """
        pass


class LeafDependencyGraphFactory(DependencyGraphFactory):
    # Inherit docstring from DependencyGraphFactory
    def subscribe_composite_trackers(self):
        system: LeafSystem = self.system

        def _add_trackers(
            source_tickets,
            subscriber_ticket,
            description_prefix,
        ):
            for index, ticket in enumerate(source_tickets):
                subscriber = self.dependency_graph[subscriber_ticket]
                source_tracker = DependencyTracker(
                    description=f"{description_prefix} {index}",
                    ticket=ticket,
                    system_id=system.system_id,
                )
                # logger.debug(
                #     f"Adding {description_prefix} {source_tracker} to {subscriber}"
                # )
                subscriber.add_prerequisite(source_tracker)

        # Discrete state trackers
        for i, ticket in enumerate(system._discrete_state_tickets):
            self.dependency_graph[ticket] = DependencyTracker(
                f"{system.name}:xd[{i}]",
                ticket,
                system_id=system.system_id,
            )
        _add_trackers(
            system._discrete_state_tickets,
            DependencyTicket.xd,
            "discrete state group",
        )

        # Parameter trackers
        for name, ticket in system._parameter_tickets.items():
            self.dependency_graph[ticket] = DependencyTracker(
                f"{system.name}:{name}",
                ticket,
                system_id=system.system_id,
            )
        _add_trackers(
            system._parameter_tickets,
            DependencyTicket.p,
            "numeric parameter group",
        )

        # Now loop again and subscribe to all prerequisites
        #   Doing this separately ensures that, for instance, port inputs can be
        #   added as dependencies for port outputs in feedthrough blocks.
        for source in system.cache_sources:
            tracker = self.dependency_graph[source.ticket]
            for prerequisite in source.prerequisites_of_calc:
                # logger.debug(
                #     f"Adding prerequisite {prerequisite} to {tracker} for {source.description}"
                # )
                tracker.add_prerequisite(self.dependency_graph[prerequisite])


class DiagramDependencyGraphFactory(DependencyGraphFactory):
    # Inherit docstring from DependencyGraphFactory
    def subscribe_composite_trackers(self):
        system: Diagram = self.system
        composites = [
            DependencyTicket.xd,
            DependencyTicket.mode,
            DependencyTicket.p,
            DependencyTicket.xcdot,
        ]

        # Subscribe "composites" to subsystem trackers
        for subsystem in system.nodes:
            # logger.debug(f"Subscribing composites to subsystem {subcontext.name}")
            for ticket in composites:
                sub_tracker = subsystem.dependency_graph[ticket]
                self.dependency_graph[ticket].add_prerequisite(sub_tracker)

    # Inherit docstring from DependencyGraphFactory
    def subscribe_ports(self):
        system: Diagram = self.system

        # Connect input/output ports within the diagram
        for dst, src in system.connection_map.items():
            dst_system, iport_index = dst
            src_system, oport_index = src

            iport_ticket = dst_system.input_ports[iport_index].ticket
            oport_ticket = src_system.output_ports[oport_index].ticket

            iport_tracker = dst_system.dependency_graph[iport_ticket]
            oport_tracker = src_system.dependency_graph[oport_ticket]

            iport_tracker.add_prerequisite(oport_tracker)

        # Subscribe exported child inputs to diagram-external inputs
        for subsystem_locator in system._input_port_map:
            subsystem, subsystem_port_index = subsystem_locator

            # Retrieve the DependencyTracker for the exported child input port
            subsystem_port = subsystem.input_ports[subsystem_port_index]
            subsystem_tracker = subsystem.dependency_graph[subsystem_port.ticket]

            # Retrieve the DependencyTracker for the diagram input port
            diagram_port_index = system._input_port_map[subsystem_locator]
            diagram_port = system.input_ports[diagram_port_index]

            # Subscribe the subsystem input port to the diagram input port
            diagram_tracker = self.dependency_graph[diagram_port.ticket]
            subsystem_tracker.add_prerequisite(diagram_tracker)

        # Subscribe diagram-external outputs to exported child outputs
        for subsystem_locator in system._output_port_map:
            subsystem, subsystem_port_index = subsystem_locator

            # Retrieve the DependencyTracker for the exported child output port
            subsystem_port = subsystem.output_ports[subsystem_port_index]
            subsystem_tracker = subsystem.dependency_graph[subsystem_port.ticket]

            # Retrieve the DependencyTracker for the diagram output port
            diagram_port_index = system._output_port_map[subsystem_locator]
            diagram_port = system.output_ports[diagram_port_index]

            # Subscribe the diagram output port to the subsystem output port
            diagram_tracker = self.dependency_graph[diagram_port.ticket]
            diagram_tracker.add_prerequisite(subsystem_tracker)
