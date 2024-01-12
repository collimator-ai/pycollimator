"""Classes for evaluating and storing results of calculations.

At the moment the caching is barely used (only for determining LeafSystem feedthrough
during automatic loop detection), but are preserved in case it is useful for when
function ordering replaces the current lazy evaluation model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Callable, List, Mapping
import dataclasses

from .dependency_graph import (
    next_dependency_ticket,
    DependencyTicket,
)

if TYPE_CHECKING:
    from ..math_backend.typing import Array
    from .system_base import SystemBase
    from .context import ContextBase

__all__ = [
    "Cache",
    "CacheSource",
    "CachedValue",
]


@dataclasses.dataclass
class CacheSource:
    """The system (pure function) side of a cache entry (CacheEntry in Drake).

    This stores the function that computes the value, as well as the prerequisites of
    the computation and the information necessary to get the value from the context.

    Attributes:
        system (SystemBase):
            The system that owns this cache source.
        ticket (int):
            The dependency ticket associated with this cache source.  See
            DependencyTicket for built-in tickets.
        name (str):
            A description of this cache source.
        prerequisites_of_calc (List[DependencyTicket]):
            Direct prerequisites of the computation, used for dependency tracking.
            These might be built-in tickets or tickets associated with other
            CacheSources.
        default_value (Array):
            A dummy value of the correct shape/dtype to fill in the initial cache,
            if known.  If None, any type checking will rely on propagating upstream
            information via the callback.
        cache_index (int):
            The index of this cache source in the system's list of cache sources.
    """

    callback: dataclasses.InitVar[Callable[[ContextBase], Array]]
    system: SystemBase
    cache_index: int
    ticket: DependencyTicket = None
    name: str = None
    prerequisites_of_calc: List[DependencyTicket] = None
    default_value: Array = None

    def __post_init__(self, callback):
        self._callback = callback  # Given root context, return calculated value

        if self.ticket is None:
            self.ticket = next_dependency_ticket()
        assert isinstance(self.ticket, int)

        if self.prerequisites_of_calc is None:
            self.prerequisites_of_calc = []

    def __hash__(self) -> int:
        locator = (self.system, self.cache_index)
        return hash(locator)

    def __repr__(self) -> str:
        return f"{self.name}(ticket = {self.ticket})"

    def eval(self, root_context: ContextBase) -> Array:
        """Evaluate the cache source and return the calculated value.

        Args:
            root_context: The root context used for the evaluation.

        Returns:
            The calculated value from the callback, expected to be a Array.
        """
        return self._callback(root_context)

    @property
    def has_default_prerequisites(self) -> bool:
        """Check if the cache source has default prerequisites (all_sources)

        Returns:
            True if the cache source depends only on DependencyTicket.all_sources
        """
        return (
            len(self.prerequisites_of_calc) == 1
            and DependencyTicket.all_sources in self.prerequisites_of_calc
        )


class CachedValue(NamedTuple):
    """The context (state) side of the cache entry (CacheEntryValue in Drake).

    This is the object that contains the actual data/value computed by a CacheSource,
    as well as tracking whether or not it is up to date.

    Attributes:
        ticket (int):
            The dependency ticket associated with this cache source.
        value (Array):
            The cached value resulting from evaluating the CacheSource callback.
        is_out_of_date (bool):
            Flag indicating whether the cached value is out of date.
    """

    ticket: int
    value: Array = None
    is_out_of_date: bool = True

    def mark_up_to_date(self) -> CachedValue:
        """Mark the cached value as up to date.

        Returns:
            A new CachedValue object with `is_out_of_date` set to False.
        """
        return self._replace(is_out_of_date=False)  # pylint: disable=no-member

    def mark_out_of_date(self) -> CachedValue:
        """Mark the cached value as out of date.

        Returns:
            A new CachedValue object with `is_out_of_date` set to True.
        """
        return self._replace(is_out_of_date=True)  # pylint: disable=no-member

    @property
    def is_initialized(self) -> bool:
        """Check if the cached value is initialized.

        The cached value is considered to be "initialized" if it holds a value
        that is not None.  If the value is None, this indicates that no default_value
        was provided for the CacheSource, and the value must be initialized by
        evaluating the CacheSource callback (propagating the information from upstream)

        Returns:
            True if the cached value is initialized, False otherwise.
        """
        return self.value is not None


Cache = Mapping[int, CachedValue]
