"""Container classes for state information.

The only actual class definition is for LeafState, which contains:
- Continuous state: the component of state that evolves in continuous time
- Discrete state: component of the state that does not change in continuous time. Note
    That this does not necessarily take on discrete _values_ (though it could).
- Mode: an integer value indicating the current "mode", "stage", or discrete-valued
    state component of the system.  Used for finite state machines or multi-stage
    hybrid systems.

Diagram-level state information is stored in containers of LeafStates as defined
in the type hints towards the end of the file.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Hashable, Mapping
import dataclasses

from ..math_backend import reshape
from jax import tree_util

if TYPE_CHECKING:
    from ..math_backend.typing import Array

__all__ = [
    "State",
    "LeafState",
    "ContinuousState",
    "DiscreteState",
    "Mode",
]


@dataclasses.dataclass(frozen=True)
class LeafState:
    """Container for state information for a leaf system.

    Attributes:
        name (str):
            Name of the leaf system that owns this state.
        continuous_state (Array):
            Continuous state of the system, i.e. the component of state that evolves in
            continuous time. If the system has no continuous state, this will be None.
        discrete_state (List[Array]):
            Discrete state of the system, i.e. one or more components of state that do
            not change continuously with ime (not necessarily discrete-_valued_). If
            the system has no discrete state, this will be None.
        mode (int):
            An integer value indicating the current "mode", "stage", or discrete-valued
            state component of the system.  Used for finite state machines or
            multi-stage hybrid systems.  If the system has no mode, this will be None.

    Notes:
        (1) This class is immutable.  To modify a LeafState, use the `with_*` methods.

        (2) Although the state components have type hints corresponding to their
        standard usage, in fact the only restriction is that they be JAX pytrees. That
        is, they may be nested containers (tuples, namedtuples, lists, dicts, etc.) of
        arrays.  This allows for more flexibility in defining state components, e.g.
        a second-order system may define a namedtuple of generalized coordinates and
        velocities rather than concatenating into a single array.
    """

    name: str = None
    continuous_state: Array = None
    discrete_state: List[Array] = None
    mode: int = None

    def __repr__(self) -> str:
        states = []
        if self.continuous_state is not None:
            states.append(f"xc={self.continuous_state}")
        if self.discrete_state is not None:
            states.append(f"xd={self.discrete_state}")
        states.append(f"s={self.mode}")
        return f"{type(self).__name__}({', '.join(states)})"

    def with_continuous_state(self, value: Array) -> LeafState:
        """Create a copy of this LeafState with the continuous state replaced."""
        if value is not None and self.continuous_state is not None:
            value = reshape(value, self.continuous_state.shape)

        return dataclasses.replace(self, continuous_state=value)

    @property
    def num_continuous_states(self) -> int:
        return self.continuous_state.size if self.continuous_state is not None else 0

    @property
    def has_continuous_state(self) -> bool:
        return self.num_continuous_states > 0

    def with_discrete_state(self, value: List[Array]) -> LeafState:
        """Create a copy of this LeafState with the discrete state replaced."""
        return dataclasses.replace(self, discrete_state=value)

    @property
    def num_discrete_states(self) -> int:
        return len(self.discrete_state) if self.discrete_state is not None else 0

    @property
    def has_discrete_state(self) -> bool:
        return self.num_discrete_states > 0

    def with_mode(self, value: int) -> LeafState:
        """Create a copy of this LeafState with the mode replaced."""
        return dataclasses.replace(self, mode=value)

    @property
    def has_mode(self) -> bool:
        return self.mode is not None


#
# Type hints for export
#
# These are the union of the leaf state types and what would be the corresponding
# diagram-level data types.  These should be used for type hints in base classes,
# but the concrete types should be preferred in implementations where possible.
if TYPE_CHECKING:
    State = Union[LeafState, Mapping[Hashable, LeafState]]
    ContinuousState = Union[Array, List[Array]]
    DiscreteState = Union[List[Array], List[List[Array]]]
    Mode = Union[int, List[int]]


#
# Register as custom pytree nodes
#    https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
#
# Even though this is a JAX utility, it can also be useful for generally working
# with tree-structured data, so we can also use these with other math backends.
#
def _leaf_state_flatten(state: LeafState):
    children = (
        state.continuous_state,
        state.discrete_state,
        state.mode,
    )
    aux_data = (state.name,)
    return children, aux_data


def _leaf_state_unflatten(aux_data, children):
    return LeafState(
        name=aux_data[0],
        continuous_state=children[0],
        discrete_state=children[1],
        mode=children[2],
    )


tree_util.register_pytree_node(
    LeafState,
    _leaf_state_flatten,
    _leaf_state_unflatten,
)
