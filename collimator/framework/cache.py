# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

"""Classes for evaluating and storing results of calculations.

The SystemCallback class is a mechanism for associating dependencies with a function
defined for a particular system.  This can include ports, event update functions, the
right-hand-side of an ODE, etc.  Declaring these functions as SystemCallbacks will
automatically create the necessary dependency tracking infrastructure to construct and
sort the execution graphs.

At the moment the caching is barely used (only for determining LeafSystem feedthrough
during automatic loop detection), but are preserved in case it is useful for when
function ordering replaces the current lazy evaluation model.  In this case the results
of each SystemCallback (e.g. output port evaluation) can be stored in the cache and
retrieved by other SystemCallbacks that depend on them.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple, Callable, List
import dataclasses

from .dependency_graph import (
    next_dependency_ticket,
    DependencyTicket,
)

from collimator.logging import logger
from collimator.backend import numpy_api as cnp
from collimator.framework.error import CallbackIsNotDifferentiableError

if TYPE_CHECKING:
    from ..backend.typing import Array
    from .dependency_graph import DependencyTracker
    from .system_base import SystemBase
    from .context import ContextBase
    from .event import Event

__all__ = [
    "SystemCallback",
    "CallbackTracer",
]


@dataclasses.dataclass
class SystemCallback:
    """A function associated with a system that has has specified dependencies.

    This can include port update rules, discrete update functions, the right-hand-side
    of an ODE, etc. Storing these functions as SystemCallbacks allows the system, or a
    Diagram containing the system, to track dependencies across the system or diagram.

    Attributes:
        system (SystemBase):
            The system that owns this callback.
        ticket (int):
            The dependency ticket associated with this callback.  See DependencyTicket
            for built-in tickets. If None, a new ticket will be generated.
        name (str):
            A short description of this callback function.
        prerequisites_of_calc (List[DependencyTicket]):
            Direct prerequisites of the computation, used for dependency tracking.
            These might be built-in tickets or tickets associated with other
            SystemCallbacks.
        default_value (Array):
            A dummy value of the same shape/dtype as the result, if known.  If None,
            any type checking will rely on propagating upstream information via the
            callback.
        callback_index (int):
            The index of this function in the system's list of associated callbacks.
        event (Event):
            Optionally, the callback function may be associated with an event.  If so,
            the associated trackers can be used to sort event execution order in addition
            to the regular callback execution order. For example, if an OutputPort is of
            sample-and-hold type, then this will be the event that periodically updates
            the output value. Default is None.
    """

    callback: dataclasses.InitVar[Callable[[ContextBase], Array]]
    system: SystemBase
    callback_index: int
    ticket: DependencyTicket = None
    name: str = None
    prerequisites_of_calc: List[DependencyTicket] = None
    default_value: Array = None
    event: Event = None

    # If the result is cached (e.g. an output port of "sample-and-hold" type),
    # this will be the index of the cache in the system's cache list.
    cache_index: int = None

    def __post_init__(self, callback):
        self._callback = callback  # Given root context, return calculated value

        if self.ticket is None:
            self.ticket = next_dependency_ticket()
        assert isinstance(self.ticket, int)

        if self.prerequisites_of_calc is None:
            self.prerequisites_of_calc = []

        logger.debug(
            "Initialized callback %s:%s with prereqs %s",
            self.system.name_path_str,
            self.name,
            self.prerequisites_of_calc,
        )

        # A basic port output data cache for numpy
        self._basic_output_cache = BasicOutputCache(self)

    def __hash__(self) -> int:
        locator = (self.system, self.callback_index)
        return hash(locator)

    def __repr__(self) -> str:
        return f"{self.name}(ticket = {self.ticket})"

    def eval(self, root_context: ContextBase) -> Array:
        """Evaluate the callback function and return the calculated value.

        Args:
            root_context: The root context used for the evaluation.

        Returns:
            The calculated value from the callback, expected to be a Array.
        """
        if not root_context.is_initialized:
            if self.default_value is None:
                self.default_value = self._callback(root_context)
            return self.default_value

        # Note: using the BasicOutputCache here does not give any performance
        # gain, due to the overhead of computing the keys and lookups.
        try:
            result = self._callback(root_context)
        except ValueError as e:
            # this error is raised if the callback is not differentiable
            if "do not support JVP." in str(e):
                raise CallbackIsNotDifferentiableError(
                    system=self.system,
                    port_name=self.name,
                )
            raise
        return result

    @property
    def tracker(self) -> DependencyTracker:
        return self.system.dependency_graph[self.ticket]


class CallbackTracer(NamedTuple):
    """A stand-in for a value in the computation graph.

    The purpose of this class is to track whether a value is modified by the various
    computations in a system.  This is used for automatically determining feedthrough
    port pairs in a LeafSystem.

    Attributes:
        ticket (int):
            The dependency ticket associated with the result of this callback. Mainly
            useful for debugging.
        is_out_of_date (bool):
            Flag indicating whether the value is out of date as a result of upstream
            prerequisite values becoming out of date.
    """

    ticket: int
    is_out_of_date: bool = True

    def mark_up_to_date(self) -> CallbackTracer:
        """Mark the value as up to date.

        Returns:
            A new CallbackTracer object with `is_out_of_date` set to False.
        """
        return self._replace(is_out_of_date=False)  # pylint: disable=no-member

    def mark_out_of_date(self) -> CallbackTracer:
        """Mark the value as out of date.

        Returns:
            A new CallbackTracer object with `is_out_of_date` set to True.
        """
        return self._replace(is_out_of_date=True)  # pylint: disable=no-member


class BasicOutputCache:
    """
    A very simple quite "hacky" implementation of a signals output cache.

    This is meant for numpy backend only, can not be expected to work with JAX,
    even in check_types because arrays should be traceable.
    """

    __active = False

    # This env var might be temporary while we test the implementation, or until
    # we have a proper solution that solves it for JAX and numpy both.
    __is_disabled = os.environ.get("COLLIMATOR_DISABLE_OUTPUT_CACHE", "0") == "1"
    __global_key = 0

    @classmethod
    def activate(cls, backend: str = None):
        """Enables caching conditionally.

        Enables if using numpy backend.
        Calling this function invalidates all caches.
        """
        cls.__active = (not cls.__is_disabled) and (
            backend or cnp.active_backend
        ) == "numpy"
        cls.__global_key += 1

    @classmethod
    def is_active(cls):
        return cls.__active and not cls.__is_disabled

    def __init__(self, target: SystemBase | SystemCallback):
        self._key = None
        self._value = None
        self._global_key_at_creation = self.__global_key
        # if hasattr(target, "system"):
        #     self._debug_name = f"{target.system.name_path_str}.{target.name}"
        # else:
        #     self._debug_name = target.name_path_str

    def _cache_key(self, context):
        # This uses a pretty poor hashing mechanism to validate that a given
        # cache entry is still valid. id() will be reused (it's the C pointer)
        # and time may not change during a solver reset. Mainly, we rely on
        # manual calls to cache invalidation. NOTE: only for numpy as of now.
        return (
            self._global_key_at_creation
            ^ id(context)
            ^ id(context.parameters)
            ^ hash(context.time)
        )

    def invalidate(self):
        self._key = None
        self._value = None
        self._global_key_at_creation = self.__global_key

    def get(self, context):
        if not self.is_active() or self._key != self._cache_key(context):
            return None
        return self._value

    def set(self, context, value):
        if not self.is_active():
            return
        self._key = self._cache_key(context)
        self._value = value
