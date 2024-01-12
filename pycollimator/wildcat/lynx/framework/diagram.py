"""Composite block-diagram representation of a dynamical system.

A Diagram is a collection of Systems connected together to form a larger hybrid
dynamical system. Diagrams can be nested to any depth, creating a tree-structured
block diagram.

As with the LeafSystem, the Diagram comprises a a collection of pure functions that
can be evaluated given a corresponding Context object containing the composite
states, parameters, etc.
"""
from __future__ import annotations
import dataclasses
from typing import (
    TYPE_CHECKING,
    List,
    Mapping,
    Set,
    Tuple,
    Hashable,
    Iterator,
)
from collections import OrderedDict

from ..logging import logger
from .event import DiagramCompositeEventCollection
from .system_base import SystemBase, UpstreamEvalError
from .context_factory import DiagramContextFactory
from .dependency_graph import DiagramDependencyGraphFactory


__all__ = [
    "Diagram",
]

if TYPE_CHECKING:
    from .port import InputPortLocator, OutputPortLocator
    from .state import LeafState
    from .leaf_system import LeafSystem
    from .context import DiagramContext
    from ..math_backend.typing import Array


@dataclasses.dataclass
class Diagram(SystemBase):
    """Composite block-diagram representation of a dynamical system.

    A Diagram is a collection of Systems connected together to form a larger hybrid
    dynamical system. Diagrams can be nested to any depth, creating a tree-structured
    block diagram.

    NOTE: The Diagram class is not intended to be constructed directly.  Instead,
    use the `DiagramBuilder` to construct a Diagram, which will pass the appropriate
    information to this constructor.
    """

    # Redefine here to make pylint happy
    name: str = None
    system_id: Hashable = None

    # None of these attributes are intended to be modified or accessed directly after
    # construction.  Instead, use the interface defined by `SystemBase`.

    # Direct children of this Diagram
    nodes: List[SystemBase] = dataclasses.field(default_factory=list)

    # Mapping from input ports to output ports of child subsystems
    connection_map: Mapping[InputPortLocator, OutputPortLocator] = dataclasses.field(
        default_factory=dict,
    )

    # Optional identifier for "reference diagrams"
    ref_id: str = None

    # Have to redeclare these here because of the way dataclasses work with ClassVar.
    # See SystemBase for documentation.
    # TODO: Deprecate these - in the future, untraced functions should be
    # called with `jax.pure_callback` rather than turning off tracing altogether.
    # "Global" tracing should be toggled at the Simulator level.
    enable_trace_cache_sources: bool = True
    enable_trace_time_derivatives: bool = True
    enable_trace_discrete_updates: bool = True
    enable_trace_unrestricted_updates: bool = True

    def __repr__(self) -> str:
        return f"{type(self).__name__}: {self.nodes}"

    def _pprint(self, prefix="") -> str:
        return f"{prefix}|-- {type(self).__name__}\n"

    def _pprint_helper(self, prefix="") -> str:
        repr = self._pprint(prefix=prefix)
        for i, substate in enumerate(self.nodes):
            repr += f"{substate._pprint_helper(prefix=f'{prefix}    ')}"
        return repr

    def __hash__(self) -> Hashable:
        return hash(self.system_id)

    def __getitem__(self, name: str) -> SystemBase:
        # Access by name - for convenient user interface only.  Programmatic
        #  access should use the nodes directly, e.g. `self.nodes[idx]`
        lookup = {node.name: node for node in self.nodes}
        return lookup[name]

    def __iter__(self) -> Iterator[SystemBase]:
        return iter(self.nodes)

    def __post_init__(self):
        super().__post_init__()

        # Set parent for all the immediate child systems
        for node in self.nodes:
            node.parent = self

        # The map of subsystem inputs/outputs to inputs/outputs of this Diagram.
        self._input_port_map: Mapping[InputPortLocator, int] = {}
        self._output_port_map: Mapping[OutputPortLocator, int] = {}

        # Also need the inverse output map, for determining feedthrough paths.
        self._inv_output_port_map: Mapping[int, OutputPortLocator] = {}

        # Leaves of the system tree (not necessarily the same as the direct
        # children of this Diagram, which may themselves be Diagrams)
        self.leaf_systems: List[LeafSystem] = []
        for sys in self.nodes:
            if isinstance(sys, Diagram):
                self.leaf_systems.extend(sys.leaf_systems)
                # No longer need the child leaf systems, since methods using this
                # should only be called from the top level.
                sys.leaf_systems = None
            else:
                self.leaf_systems.append(sys)

    @property
    def num_systems(self) -> int:
        # Number of subsystems _at this level_
        return len(self.nodes)

    def check_types(self, context: DiagramContext) -> None:
        """Perform any system-specific static analysis."""
        for system in self.nodes:
            system.check_types(context)

    #
    # Simulation interface
    #

    # Inherits docstrings from SystemBase
    def eval_time_derivatives(self, root_context: DiagramContext) -> List[Array]:
        leaf_systems = [
            subctx.owning_system for subctx in root_context.continuous_subcontexts
        ]
        return [sys.eval_time_derivatives(root_context) for sys in leaf_systems]

    #
    # Event handling
    #
    @property
    def periodic_events(self) -> DiagramCompositeEventCollection:
        assert (
            self.leaf_systems is not None
        ), f"Can only get periodic events from top-level Diagram, not {self.system_id} with parent {self.parent.system_id}"
        events = DiagramCompositeEventCollection(
            OrderedDict(
                {sys.system_id: sys.periodic_events for sys in self.leaf_systems}
            )
        )
        return events

    @property
    def zero_crossing_events(self) -> DiagramCompositeEventCollection:
        assert (
            self.leaf_systems is not None
        ), f"Can only get zero-crossing events from top-level Diagram, not {self.system_id} with parent {self.parent.system_id}"
        return DiagramCompositeEventCollection(
            OrderedDict(
                {sys.system_id: sys.zero_crossing_events for sys in self.leaf_systems}
            )
        )

    # Inherits docstrings from SystemBase
    def determine_active_guards(
        self, root_context: DiagramContext
    ) -> DiagramCompositeEventCollection:
        assert (
            self.leaf_systems is not None
        ), f"Can only get active guards from top-level Diagram, not {self.system_id} with parent {self.parent.system_id}"
        return DiagramCompositeEventCollection(
            OrderedDict(
                {
                    sys.system_id: sys.determine_active_guards(root_context)
                    for sys in self.leaf_systems
                }
            )
        )

    # Inherits docstrings from SystemBase
    def eval_discrete_updates(
        self, root_context: DiagramContext, events: DiagramCompositeEventCollection
    ) -> List[Array]:
        leaf_systems = [
            subctx.owning_system for subctx in root_context.discrete_subcontexts
        ]
        return [sys.eval_discrete_updates(root_context, events) for sys in leaf_systems]

    # Inherits docstrings from SystemBase
    def eval_zero_crossing_updates(
        self,
        root_context: DiagramContext,
        events: DiagramCompositeEventCollection,
    ) -> dict[Hashable, LeafState]:
        substates = OrderedDict()
        for system_id, subctx in root_context.subcontexts.items():
            sys = subctx.owning_system
            substates[system_id] = sys.eval_zero_crossing_updates(root_context, events)

        return substates

    # Inherits docstrings from SystemBase
    # TODO: This should be made obsolete following WC-120. Added as part of WC-80 to
    # remove the need for passing "handler" functions, which makes the code much
    # more straightforward.
    def eval_unrestricted_updates(
        self,
        root_context: DiagramContext,
        events: DiagramCompositeEventCollection,
    ) -> dict[Hashable, LeafState]:
        substates = OrderedDict()
        for system_id, subctx in root_context.subcontexts.items():
            sys = subctx.owning_system
            substates[system_id] = sys.eval_unrestricted_updates(root_context, events)

        return substates

    #
    # I/O ports
    #
    @property
    def exported_input_ports(self):
        return self._input_port_map

    @property
    def exported_output_ports(self):
        return self._output_port_map

    def eval_subsystem_input_port(
        self, context: DiagramContext, port_locator: InputPortLocator
    ) -> Array:
        """Evaluate the input port for a child of this system given the root context.

        Args:
            context (ContextBase): root context for this system
            port_locator (InputPortLocator): tuple of (system, port_index) identifying
                the input port to evaluate

        Returns:
            Array: Value returned from evaluating the subsystem port.
        """

        is_exported = port_locator in self._input_port_map
        if is_exported:
            # The upstream source is an input to this whole Diagram; evaluate that
            # input port and use the result as the value for this one.
            port_index = self._input_port_map[port_locator]  # Diagram-level index
            return self.input_ports[port_index].eval(context)  # Return upstream value

        is_connected = port_locator in self.connection_map
        if is_connected:
            # The upstream source is an output port of one of this Diagram's child
            # subsystems; evaluate the upstream output.
            upstream_locator = self.connection_map[port_locator]

            # This will return the value of the upstream port
            return self.eval_subsystem_output_port(context, upstream_locator)

        raise RuntimeError(
            f"Shouldn't get here: port {port_locator} may not be connected"
        )

    def eval_subsystem_output_port(
        self, context: DiagramContext, port_locator: OutputPortLocator
    ) -> Array:
        """ "Evaluate the output port for a child of this system given the root context.

        Args:
            context (ContextBase): root context for this system
            port_locator (OutputPortLocator): tuple of (system, port_index) identifying
                the output port to evaluate

        Returns:
            Array: Value returned from evaluating the subsystem port.
        """
        system, port_index = port_locator
        port = system.output_ports[port_index]

        # During simulation all we should need to do is evaluate the port.
        if context.is_initialized:
            return port.eval(context)

        # If the context is not initialized, we have to determine the signal data type.
        # In the easy case, the port has a default value, so we can just use that.
        if port.default_value is not None:
            logger.debug(
                f"Using default output value of {port.default_value} for "
                f"{port_locator[0].name}"
            )
            return port.default_value

        logger.debug(
            f"Evaluating output port {port_locator} for system {port_locator[0].name}."
            f"  Context initialized? {context.is_initialized}"
        )

        # We can also allow blocks to do custom initialization, for example by looking
        # at the type of its inputs (e.g. ZeroOrderHold or DerivativeDiscrete).
        system, _ = port_locator
        context = system.initialize_static_data(context)

        # Try again to evaluate the port
        val = port.eval(context)
        logger.debug(f"  ---> {(port_locator[0].name, port_locator[1])} returns {val}")

        # If there is still no value, the port is not connected to anything.
        # Post-initialization this would be an error, but pre-initialization
        # it may be the case that the upstream is an exported input port of
        # the Diagram, so we can defer evaluation. Expect the block that is
        # doing this to handle the UpstreamEvalError appropriately.
        if val is None:
            logger.debug(
                f"Upstream evaluation of {port_locator} returned None."
                "  Deferring evaluation."
            )
            raise UpstreamEvalError(port_locator)
        return val

    #
    # System-level declarations (should be done via DiagramBuilder)
    #
    def export_input(self, locator: InputPortLocator, port_name: str) -> int:
        """Export a subsystem input port as a diagram-level input.

        This should typically only be called during construction by DiagramBuilder.
        The standard workflow will be to call export_input on the _builder_ object,
        which will automatically call this method on the Diagram once created.

        Args:
            locator (InputPortLocator): tuple of (system, port_index) identifying
                the input port to export
            port_name (str): name of the new exported input port

        Returns:
            int: index of the exported input port in the diagram input_ports list
        """
        diagram_port_index = self.declare_input_port(name=port_name)
        self._input_port_map[locator] = diagram_port_index
        return diagram_port_index

    def export_output(self, locator: OutputPortLocator, port_name: str) -> int:
        """Export a subsystem output port as a diagram-level output.

        This should typically only be called during construction by DiagramBuilder.
        The standard workflow will be to call export_input on the _builder_ object,
        which will automatically call this method on the Diagram once created.

        Args:
            locator (OutputPortLocator): tuple of (system, port_index) identifying
                the output port to export
            port_name (str): name of the new exported output port

        Returns:
            int: index of the exported output port in the diagram output_ports list
        """
        subsystem, subsystem_port_index = locator
        source_port = subsystem.output_ports[subsystem_port_index]
        diagram_port_index = self.declare_output_port(
            source_port.eval,
            name=port_name,
            prerequisites_of_calc=[source_port.ticket],
        )
        self._output_port_map[locator] = diagram_port_index
        self._inv_output_port_map[diagram_port_index] = locator
        return diagram_port_index

    #
    # Initialization
    #
    @property
    def context_factory(self) -> DiagramContextFactory:
        return DiagramContextFactory(self)

    @property
    def dependency_graph_factory(self) -> DiagramDependencyGraphFactory:
        return DiagramDependencyGraphFactory(self)

    def _has_feedthrough(self, input_port_index: int, output_port_index: int) -> bool:
        """Check if there is a direct-feedthrough path from the input port to the output port.

        Internal function used by `get_feedthrough`.  Should not typically need to
        be called directly.
        """
        # TODO: Would this be simpler if the input port map was inverted?
        input_ids = []
        for locator, index in self._input_port_map.items():
            if index == input_port_index:
                input_ids.append(locator)

        input_ids = set(input_ids)

        # Search graph for a direct-feedthrough connection from the output_port
        # to the input_port.  Maintain a set of the output port identifiers that
        # are known to have a direct-feedthrough path to the output_port
        active_set: Set[OutputPortLocator] = set()
        active_set.add(self._inv_output_port_map[output_port_index])

        while len(active_set) > 0:
            sys, sys_output = active_set.pop()
            for u, v in sys.get_feedthrough():
                if v == sys_output:
                    curr_input_id = (sys, u)
                    if curr_input_id in input_ids:
                        # Found a direct-feedthrough path to the input_port
                        return True
                    elif curr_input_id in self.connection_map:
                        # Intermediate input port has a direct-feedthrough path to
                        # output_port. Add the upstream output port (if there
                        # is one) to the active set.
                        active_set.add(self.connection_map[curr_input_id])

        # If there are no intermediate output ports with a direct-feedthrough path
        # to the output port, there is no direct feedthrough from the input port
        return False

    # Inherits docstring from SystemBase.get_feedthrough
    def get_feedthrough(self) -> List[Tuple[int, int]]:
        if self.feedthrough_pairs is not None:
            return self.feedthrough_pairs

        pairs = []
        for u in range(self.num_input_ports):
            for v in range(self.num_output_ports):
                if self._has_feedthrough(u, v):
                    pairs.append((u, v))

        self.feedthrough_pairs = pairs
        return self.feedthrough_pairs
