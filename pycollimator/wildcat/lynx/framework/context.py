"""Context classes for storing 'dynamic' data for the system (state, parameters, etc).

These should be created using ContextFactory classes by calling
`system.create_context()`. All contexts are immutable, so any changes should be made by
calling `context.with_*`, which will return a new context with the updated value.

DiagramContexts have the same basic interface as leaf contexts (e.g. `context.state`),
but they can also be indexed using system IDs to get the corresponding LeafContext.
This dict-like access is 'flat', meaning that all leaf contexts are at the same level,
regardless of their depth in the associated Diagram.

For example, if you have a diagram structured like

root[system_id=0]
|-- submodel[system_id=1]
|   |-- leaf1[system_id=2]
|   |-- leaf2[system_id=3]
|-- leaf3[system_id=4]

then the DiagramContext can be indexed with any of the _leaf_ system IDs (2, 3, or 4),
and the result will be a LeafContext. The intermediate submodel (1) does not have a
context of its own, since it cannot have any state or parameters.

In order to update an attribute of a leaf system that is a child of a diagram, you must
first get the LeafContext for that system, update the attribute, and then update the
DiagramContext with the new LeafContext using `with_subcontext`. For example:

```python
# Get the LeafContext for leaf1 from the root DiagramContext
leaf_context = root_context[2]

# Update the continuous state of leaf1
leaf_context = leaf_context.with_continuous_state(new_value)

# Update the root DiagramContext with the new LeafContext
root_context = root_context.with_subcontext(2, leaf_context)
```
"""
from __future__ import annotations
import abc

from typing import TYPE_CHECKING, List, Hashable, Mapping
from collections import OrderedDict
import dataclasses

from jax import tree_util

if TYPE_CHECKING:
    from ..math_backend.typing import (
        Array,  # Array (or possibly, nested containers of arrays)
        Scalar,  # float, int, or array with `shape=()`
    )
    from .system_base import SystemBase
    from .cache import CachedValue
    from .state import (
        LeafState,
        State,
        ContinuousState,
        DiscreteState,
        Mode,
    )


__all__ = [
    "ContextBase",
    "LeafContext",
    "DiagramContext",
]


@dataclasses.dataclass(frozen=True)
class ContextBase(metaclass=abc.ABCMeta):
    """Context object containing state, parameters, etc for a system.

    NOTE: Type hints in ContextBase indicate the union between what would be returned
    by a LeafContext and a DiagramContext. See type hints of the subclasses for
    the specific argument and return types.

    Attributes:
        owning_system (SystemBase):
            The owning system of the context.
        time (Scalar):
            The time associated with the context. Will be None unless the context
            is the root context.
        cache (Mapping[int, CachedValue]):
            Cache of computed values for the context. Currently this is only used
            during algebraic loop detection and then deleted, but it could be used
            for other purposes (specifically, function ordering) in the future.
        is_initialized (bool):
            Flag indicating if the context is initialized. This should only be set
            by the ContextFactory during creation.
    """

    owning_system: SystemBase
    time: Scalar = None
    cache: Mapping[int, CachedValue] = dataclasses.field(default_factory=dict)
    is_initialized: bool = False

    @abc.abstractmethod
    def __getitem__(self, key: Hashable) -> LeafContext:
        """Get the subcontext associated with the given system ID.

        For leaf contexts, this will return `self`, but the method is provided
        so that there is a consistent interface for working with either an
        individual LeafSystem or tree-structured Diagram.

        For nested diagrams, intermediate diagrams do not have associated contexts,
        so indexing will fail.
        """
        pass

    @abc.abstractmethod
    def with_subcontext(self, key: Hashable, ctx: LeafContext) -> ContextBase:
        """Create a copy of this context, replacing the specified subcontext."""
        pass

    def with_time(self, value: Scalar) -> ContextBase:
        """Create a copy of this context, replacing time with the given value.

        This should only be called on the root context, since it is expected that all
        subcontexts will have a time value of None to avoid any conflicts.
        """
        return dataclasses.replace(self, time=value)

    @abc.abstractproperty
    def state(self) -> State:
        pass

    @abc.abstractmethod
    def with_state(self, state: State) -> ContextBase:
        """Create a copy of this context, replacing the entire state."""
        pass

    @abc.abstractproperty
    def continuous_state(self) -> ContinuousState:
        pass

    @abc.abstractmethod
    def with_continuous_state(self, value: ContinuousState) -> ContextBase:
        """Create a copy of this context, replacing the continuous state."""
        pass

    @abc.abstractproperty
    def num_continuous_states(self) -> int:
        pass

    @abc.abstractproperty
    def has_continuous_state(self) -> bool:
        pass

    @abc.abstractproperty
    def discrete_state(self) -> DiscreteState:
        pass

    @abc.abstractmethod
    def with_discrete_state(self, value: DiscreteState) -> ContextBase:
        """Create a copy of this context, replacing the discrete state."""
        pass

    @abc.abstractproperty
    def num_discrete_states(self) -> int:
        pass

    @abc.abstractproperty
    def has_discrete_state(self) -> bool:
        pass

    @abc.abstractproperty
    def mode(self) -> Mode:
        pass

    @abc.abstractproperty
    def has_mode(self) -> bool:
        pass

    @abc.abstractmethod
    def with_mode(self, value: Mode) -> ContextBase:
        """Create a copy of this context, replacing the mode."""
        pass

    #
    # cache interface
    #
    def mark_initialized(self) -> ContextBase:
        return dataclasses.replace(self, is_initialized=True)

    def clear_cache(self) -> ContextBase:
        return dataclasses.replace(
            self,
            cache=self.owning_system.clear_cache(self.cache),
        )


@dataclasses.dataclass(frozen=True)
class LeafContext(ContextBase):
    state: LeafState = None
    parameters: Mapping[str, Array] = None

    @property
    def system_id(self) -> Hashable:
        return self.owning_system.system_id

    def __getitem__(self, key: Hashable) -> LeafContext:
        """Dummy indexing for compatibility with DiagramContexts, returning self."""
        assert key == self.system_id, f"Attempting to get subcontext {key} from {self}"
        return self

    def with_subcontext(self, key: Hashable, ctx: LeafContext) -> LeafContext:
        """Dummy replacement for compatibility with DiagramContexts, returning ctx."""
        assert (
            key == self.system_id
        ), f"System ID {key} does not match leaf ID {self.system_id}"
        assert (
            key == ctx.system_id
        ), f"System ID {key} does not match leaf ID {ctx.system_id}"
        return ctx

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sys={self.system_id})"

    def with_state(self, state: LeafState) -> LeafContext:
        return dataclasses.replace(self, state=state)

    @property
    def continuous_state(self) -> Array:
        return self.state.continuous_state

    def with_continuous_state(self, value: Array) -> LeafContext:
        return dataclasses.replace(self, state=self.state.with_continuous_state(value))

    @property
    def num_continuous_states(self) -> int:
        return self.state.num_continuous_states

    @property
    def has_continuous_state(self) -> bool:
        return self.state.has_continuous_state

    @property
    def discrete_state(self) -> List[Array]:
        return self.state.discrete_state

    def with_discrete_state(self, value: List[Array]) -> LeafContext:
        return dataclasses.replace(self, state=self.state.with_discrete_state(value))

    @property
    def num_discrete_states(self) -> int:
        return self.state.num_discrete_states

    @property
    def has_discrete_state(self) -> bool:
        return self.state.has_discrete_state

    @property
    def mode(self) -> int:
        return self.state.mode

    @property
    def has_mode(self) -> bool:
        return self.state.has_mode

    def with_mode(self, value: int) -> LeafContext:
        return dataclasses.replace(self, state=self.state.with_mode(value))

    def with_parameter(self, name: str, value: Array) -> LeafContext:
        """Create a copy of this context, replacing the specified parameter."""
        return dataclasses.replace(self, parameters={**self.parameters, name: value})

    def with_parameters(self, new_parameters: Mapping[str, Array]) -> LeafContext:
        """Create a copy of this context, replacing only the specified parameters."""
        return dataclasses.replace(
            self, parameters={**self.parameters, **new_parameters}
        )


@dataclasses.dataclass(frozen=True)
class DiagramContext(ContextBase):
    subcontexts: OrderedDict[Hashable, LeafContext] = dataclasses.field(
        default_factory=OrderedDict
    )

    def _check_key(self, key: Hashable) -> None:
        assert key == self.owning_system.system_id or key in self.subcontexts, (
            f"System ID {key} not found in DiagramContext {self}.\nIf this ID "
            "references an intermediate diagram, note that intermediate diagrams do "
            "not have associated contexts. Only the root diagram and leaf systems have "
            "contexts."
        )

    def __getitem__(self, key: Hashable) -> LeafContext:
        self._check_key(key)
        if key == self.owning_system.system_id:
            return self
        return self.subcontexts[key]

    def with_subcontext(self, key: Hashable, ctx: LeafContext) -> DiagramContext:
        self._check_key(key)
        subcontexts = self.subcontexts.copy()
        subcontexts[key] = ctx
        return dataclasses.replace(self, subcontexts=subcontexts)

    #
    # Simulation interface
    #
    @property
    def state(self) -> Mapping[Hashable, LeafState]:
        return OrderedDict(
            {system_id: subctx.state for system_id, subctx in self.subcontexts.items()}
        )

    @property
    def continuous_subcontexts(self) -> List[LeafContext]:
        return [
            subctx
            for subctx in self.subcontexts.values()
            if subctx.has_continuous_state
        ]

    @property
    def continuous_state(self) -> List[Array]:
        return [subctx.continuous_state for subctx in self.continuous_subcontexts]

    def with_continuous_state(self, sub_xcs: List[Array]) -> DiagramContext:
        # Shallow copy the subcontexts - only modify the ones that have continuous states
        new_subcontexts = self.subcontexts.copy()
        for subctx, sub_xc in zip(self.continuous_subcontexts, sub_xcs):
            new_subcontexts[subctx.system_id] = subctx.with_continuous_state(sub_xc)
        return dataclasses.replace(self, subcontexts=new_subcontexts)

    @property
    def num_continuous_states(self) -> int:
        return sum(
            [subctx.num_continuous_states for subctx in self.subcontexts.values()]
        )

    @property
    def has_continuous_state(self) -> bool:
        return self.num_continuous_states > 0

    @property
    def discrete_subcontexts(self) -> List[LeafContext]:
        return [
            subctx for subctx in self.subcontexts.values() if subctx.has_discrete_state
        ]

    @property
    def discrete_state(self) -> List[List[Array]]:
        return [subctx.discrete_state for subctx in self.discrete_subcontexts]

    def with_discrete_state(self, sub_xds: List[List[Array]]) -> DiagramContext:
        # Shallow copy the subcontexts - only modify the ones that have discrete states
        new_subcontexts = self.subcontexts.copy()
        for subctx, sub_xd in zip(self.discrete_subcontexts, sub_xds):
            new_subcontexts[subctx.system_id] = subctx.with_discrete_state(sub_xd)
        return dataclasses.replace(self, subcontexts=new_subcontexts)

    @property
    def num_discrete_states(self) -> int:
        return sum([subctx.num_discrete_states for subctx in self.subcontexts.values()])

    @property
    def has_discrete_state(self) -> bool:
        return self.num_discrete_states > 0

    @property
    def mode_subcontexts(self) -> List[LeafContext]:
        return [subctx for subctx in self.subcontexts.values() if subctx.has_mode]

    @property
    def mode(self) -> List[int]:
        return [subctx.mode for subctx in self.mode_subcontexts]

    def with_mode(self, sub_modes: List[int]) -> DiagramContext:
        new_subcontexts = self.subcontexts.copy()
        for subctx, sub_mode in zip(self.mode_subcontexts, sub_modes):
            new_subcontexts[subctx.system_id] = subctx.with_mode(sub_mode)
        return dataclasses.replace(self, subcontexts=new_subcontexts)

    @property
    def has_mode(self) -> bool:
        return any([subctx.has_mode for subctx in self.subcontexts.values()])

    def with_state(self, sub_states: Mapping[Hashable, LeafState]) -> DiagramContext:
        new_subcontexts = OrderedDict()
        for system_id, sub_state in sub_states.items():
            new_subcontexts[system_id] = dataclasses.replace(
                self.subcontexts[system_id], state=sub_state
            )
        return dataclasses.replace(self, subcontexts=new_subcontexts)

    #
    #  Dependency tracking
    #
    def clear_cache(self) -> DiagramContext:
        context = super().clear_cache()
        for system_id in self.subcontexts:
            context = context.with_subcontext(
                system_id, context[system_id].clear_cache()
            )
        return context


#
# Register as custom pytree nodes
#    https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
#
# Even though this is a JAX utility, it can also be useful for generally working
# with tree-structured data, so we can also use these with other math backends.
#
def _leaf_context_flatten(context: LeafContext):
    children = (
        context.state,
        context.parameters,
        context.time,
        context.cache,
    )
    aux_data = (
        context.owning_system,
        context.is_initialized,
    )
    return children, aux_data


def _leaf_context_unflatten(aux_data, children):
    owning_system, is_initialized = aux_data
    state, parameters, time, cache = children
    return LeafContext(
        owning_system=owning_system,
        is_initialized=is_initialized,
        state=state,
        parameters=parameters,
        time=time,
        cache=cache,
    )


tree_util.register_pytree_node(
    LeafContext,
    _leaf_context_flatten,
    _leaf_context_unflatten,
)


def _diagram_context_flatten(context: DiagramContext):
    keys = context.subcontexts.keys()
    vals = list(context.subcontexts.values())
    children = (
        vals,
        context.time,
        context.cache,
    )
    aux_data = (
        keys,
        context.owning_system,
        context.is_initialized,
    )
    return children, aux_data


def _diagram_context_unflatten(aux_data, children):
    keys, owning_system, is_initialized = aux_data
    vals, time, cache = children
    subcontexts = OrderedDict(zip(keys, vals))
    return DiagramContext(
        owning_system=owning_system,
        is_initialized=is_initialized,
        subcontexts=subcontexts,
        time=time,
        cache=cache,
    )


tree_util.register_pytree_node(
    DiagramContext,
    _diagram_context_flatten,
    _diagram_context_unflatten,
)
