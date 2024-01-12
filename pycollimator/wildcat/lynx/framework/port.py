"""Classes for input/output ports of systems.

These are simple wrappers around CacheSource that add a few extra properties
and methods for convenience.

TODO: This file is small enough now that it should just be merged into system_base.py
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union
import dataclasses

# TODO: See comment in InputPort.fix_value.
# from jax.errors import ConcretizationTypeError
# from jax._src.interpreters.partial_eval import DynamicJaxprTracer

from .dependency_graph import DependencyTicket
from .cache import CacheSource

if TYPE_CHECKING:
    from .system_base import SystemBase
    from ..math_backend.typing import Array

    # These are hashable objects that can be used to locate a port within a system.
    # Used primarily for tracking connectivity in diagrams.
    InputPortLocator = Tuple[SystemBase, int]  # (system, port_index)
    OutputPortLocator = Tuple[SystemBase, int]  # (system, port_index)
    EitherPortLocator = Union[InputPortLocator, OutputPortLocator]


__all__ = [
    "InputPort",
    "OutputPort",
    "InputPortLocator",
    "OutputPortLocator",
    "EitherPortLocator",
]


@dataclasses.dataclass
class PortBase(CacheSource):
    """Base class for input and output ports."""

    # Index of the port within the owning System.  For a Diagram-level port, this
    # will be the index within the Diagram, not within the LeafSystem
    # whose port was forwarded. This should not be confused with cache_index, which
    # will index into the system cache_sources, rather than (input/output)_ports.
    # This value should be automatically created by the system when the port is added.
    #
    # Should not be None, but cannot be declared as a required argument because of
    # dataclass __init__ argument ordering, so this is checked in __post_init__.
    index: int = None

    def __post_init__(self, callback):
        super().__post_init__(callback)
        assert self.index is not None, "Port index cannot be None"

    @property
    def locator(self) -> EitherPortLocator:
        return (self.system, self.index)

    def __hash__(self) -> int:
        return hash(self.locator)


class OutputPort(PortBase):
    """Output port of a system."""

    pass


class InputPort(PortBase):
    """Input port of a system."""

    is_fixed = False  # Whether the value of this port is fixed to a constant value

    def fix_value(self, value: Array):
        """Set the value of this port to a constant.

        This can be used for testing subsystems with inputs in isolation, for instance.
        In order to undo the fixed value, the system can be re-created or added to
        a Diagram and the input can be connected to a different port.

        Args:
            value (Array): The value to fix the port to.

        Notes:
            This should not be called in any "traced" code, for example during an
            optimization or differentiation pass.  This is because the value will be
            a JAX tracer that is then stored in the system, creating a side effect
            and possible a leaked tracer error.  Instead, the value should be fixed
            by creating a new diagram where the input is a Constant block that is then
            connected to the subsystem.  Then the value of the Constant block is itself
            a traced value in the context and can be updated in the traced code.
        """
        # TODO: Should check that the value is not a JAX tracer. See:
        # https://collimator.atlassian.net/browse/WC-114
        # Uncomment the following code (currently causes some tests to fail).
        # if isinstance(value, DynamicJaxprTracer):
        #     raise ConcretizationTypeError(
        #         f"Cannot fix value of port {self.name} to a traced value. "
        #         "Assigning fixed values must be done statically (i.e. before JIT "
        #         "compilation, autodifferentiation, etc.)"
        #     )
        self._callback = lambda _: value
        self.default_value = value
        self.prerequisites_of_calc = [DependencyTicket.nothing]
        self.is_fixed = True
