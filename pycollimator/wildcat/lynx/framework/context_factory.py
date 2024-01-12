"""Factory classes for generating context classes.

These should not be used directly by users.  Instead, use the `create_context`
method on `SystemBase` objects.  These classes are broken out because they contain
significant logic that is not relevant to the user-facing System API and is only used
for context creation.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from collections import OrderedDict

from .error import StaticError
from ..logging import logger
from .cache import CachedValue
from .context import LeafContext, DiagramContext

if TYPE_CHECKING:
    from ..math_backend.typing import Scalar
    from .system_base import SystemBase
    from .context import ContextBase
    from .dependency_graph import Cache
    from .diagram import Diagram
    from .leaf_system import LeafSystem


class ContextFactory(metaclass=abc.ABCMeta):
    """Abstract base class for creating context objects."""

    def __init__(self, system: SystemBase):
        """Initialize the ContextFactory with the given system.

        To actually create the context, call the ContextFactory object as a function.
        This should happen automatically when calling `system.create_context`.

        Args:
            system (SystemBase):
                The system for which the context is being created.
        """
        self.system = system

    def __call__(self, time: Scalar = 0.0, check_types: bool = True) -> ContextBase:
        """Create a context object for the system.

        Args:
            time (Scalar):
                The initial simulation time for the context.  Can also be used
                to determine the dtype of the time variable. Defaults to 0.0 (a float).
            check_types (bool):
                Whether to perform basic type checking.  Should only be set to True
                when creating the root context.

        Returns:
            The created context object.
        """
        logger.debug(f"Creating context for {self.system.name}")
        ctx = self.create_node_context()
        ctx = ctx.with_time(time)

        # At this point the call should be "untraced", meaning we can change
        # attributes of the owning system based on the context.  This is useful
        # when the system has an internal state with dtype or shape that must be
        # determined based on what it's connected to (see `DerivativeDiscrete` block
        # for an example).
        ctx = self.initialize_static_data(ctx)

        # Optionally, perform basic type checking by evaluating all the cache sources.
        # Systems are allowed to define any implementation-specific type checking.
        # Since this won't make sense on partially-constructed digrams, it should only
        # be done for the root context.
        if check_types:
            self.check_types(ctx)

        ctx = ctx.mark_initialized()
        return ctx

    @abc.abstractmethod
    def create_node_context(self) -> ContextBase:
        """Create a generic context for this system

        This will not perform any type checking or static analysis, but does initialize
        state and parameters, cache, etc.

        Returns:
            The created node context.
        """
        pass

    def initialize_static_data(self, root_context: ContextBase) -> ContextBase:
        """Perform any system-specific static analysis.

        For example, this could be used to determine the dtype of a system's internal
        state based on the types of its inputs.  This is called after the context is
        created, but before any type checking is performed.
        """
        return self.system.initialize_static_data(root_context)

    def check_types(self, root_context: ContextBase):
        """Perform basic type checking by evaluating cache sources.

        All cache sources (ports, ODE right-hand-side functions, etc) are evaluated
        to ensure that there are no shape mismatches, etc.  Sources with default values
        are skipped.  If any exceptions arise, they are caught and re-raised as a
        StaticError.

        Args:
            root_context: The root context object.

        Raises:
            StaticError: If any exceptions are raised during type checking.
        """
        system = self.system
        logger.debug(f"Cache sources: {system.cache_sources}")
        for source in system.cache_sources:
            if source.default_value is None:
                try:
                    value = source.eval(root_context)

                    if value is None:
                        logger.error(
                            f"Cache source {source} was not initialized by source.eval"
                        )

                except Exception as exc:
                    logger.error(f"Error {exc} from {system.name}")
                    raise StaticError(system) from exc

                logger.debug(
                    f"    ---> check_types for {source} returns {root_context} with value {value}: "
                )

    def create_cache(self) -> Cache:
        """
        Create a dictionary mapping cache indices to CachedValue objects.

        NOTE: Currently cache is only used to detect feedthrough pairs in LeafSystems
        by marking inport cache values as "out of date" and then seeing if the
        cache invalidation propagates to the corresponding outport. After this
        is done the cache is destroyed so that it doesn't take part in tracing
        the entire simulation. However, the cache could be used to store port
        values for a CMLC-like function evaluation scheme in the future.

        Returns:
            The created cache.
        """
        cache = {}
        for source in self.system.cache_sources:
            cache[source.cache_index] = CachedValue(
                ticket=source.ticket,
                value=source.default_value,
            )
        return cache


class LeafContextFactory(ContextFactory):
    """ContextFactory for leaf systems."""

    # Inherit docstring from ContextFactory
    def create_node_context(self) -> LeafContext:
        system: LeafSystem = self.system

        logger.debug(f"Creating cache entries for {system.cache_sources}")

        cache = self.create_cache()

        state = system.create_state()
        params = system.create_parameters()
        logger.debug(f"Created state {state} and parameters {params} for {system.name}")

        return LeafContext(
            owning_system=system,
            state=state,
            parameters=params,
            cache=cache,
        )


class DiagramContextFactory(ContextFactory):
    """ContextFactory for diagram systems."""

    # Inherit docstring from ContextFactory
    def create_node_context(self) -> DiagramContext:
        system: Diagram = self.system
        cache = self.create_cache()

        subcontext_nodes = [
            sys.context_factory.create_node_context() for sys in system.nodes
        ]

        subcontexts = OrderedDict()
        for ctx in subcontext_nodes:
            if isinstance(ctx, DiagramContext):
                subcontexts.update(ctx.subcontexts)
            else:
                subcontexts[ctx.owning_system.system_id] = ctx

        return DiagramContext(
            owning_system=system,
            subcontexts=subcontexts,
            cache=cache,
        )

    # Inherit docstring from ContextFactory
    def initialize_static_data(self, root_context: DiagramContext) -> DiagramContext:
        system: Diagram = self.system
        for sys in system.nodes:
            root_context = sys.context_factory.initialize_static_data(root_context)
        return super().initialize_static_data(root_context)

    # Inherit docstring from ContextFactory
    def check_types(self, root_context: DiagramContext):
        system: Diagram = self.system
        for sys in system.nodes:
            logger.debug(f"**** Initializing cache for {sys.name} ****")
            sys.context_factory.check_types(root_context)
        super().check_types(root_context)
