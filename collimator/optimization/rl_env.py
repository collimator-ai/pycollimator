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

"""Wrapper for training system models using reinforcement learning."""

from __future__ import annotations
import abc
from functools import partial
from typing import TYPE_CHECKING, Hashable

import numpy as np

import jax
import jax.numpy as jnp

import flax
from brax.envs.base import Env as BraxEnv, State as BraxState

import collimator
from collimator.library import Constant
from collimator.framework import DiagramBuilder

if TYPE_CHECKING:
    from collimator.framework import SystemBase, ContextBase

__all__ = ["RLEnv"]


def _wrapper_diagram(
    plant: SystemBase, act_size: int, input_name: str = "const_in"
) -> SystemBase:
    """Embed the plant into a larger diagram with a constant input."""
    builder = DiagramBuilder()
    submodel = builder.add(plant)
    const_in = builder.add(Constant(value=np.zeros(act_size), name=input_name))
    builder.connect(const_in.output_ports[0], submodel.input_ports[0])
    return builder.build()


def _prepare_context(input_id, context: ContextBase, u: jax.Array) -> ContextBase:
    """Prepare the context by updating the input port with the new value."""
    u_context = context[input_id].with_parameter("value", u)
    return context.with_subcontext(input_id, u_context)


def _forward(
    system: SystemBase,
    dt: float,
    input_id: Hashable,
    context: ContextBase,
    u: jax.Array,
) -> ContextBase:
    """Simulate the system forward in time and return the final context."""
    context = _prepare_context(input_id, context, u)
    t0 = context.time
    results = collimator.simulate(system, context, (t0, t0 + dt))
    return results.context


def _get_obs(
    plant: SystemBase, input_id: Hashable, context: ContextBase, u: jax.Array
) -> jax.Array:
    """Return the observation from the plant given the current context."""
    context = _prepare_context(input_id, context, u)
    return plant.output_ports[0].eval(context)


# Dummy subclass of brax.envs.base.State to distinguish from collimator.State
# This uses flax.struct.dataclass for consistency with brax, although this is
# different from how we define custom pytrees in the rest of collimator.
@flax.struct.dataclass
class RLState(BraxState):
    pass


class RLEnv(BraxEnv):
    """Base class for reinforcement learning environments in Collimator."""

    def __init__(self, plant: SystemBase, act_size: int, dt: float):
        if len(plant.input_ports) != 1:
            raise ValueError("Plant must have exactly one input port.")

        if len(plant.output_ports) != 1:
            raise ValueError("Plant must have exactly one output port.")

        self._act_size = act_size
        self._plant = plant
        self._plant_id = plant.system_id
        self.dt = dt

        # Embed the plant within a simple wrapper diagram that has a constant input
        input_name = "const_in"
        self.diagram = _wrapper_diagram(plant, act_size, input_name=input_name)
        input_id = self.diagram[input_name].system_id

        self._forward = partial(_forward, self.diagram, dt, input_id)
        self._get_obs = partial(_get_obs, plant, input_id)

        self._static_context = self.diagram.create_context()
        obs = self._get_obs(self._static_context, jnp.zeros(self._act_size))
        self._obs_size = obs.size

    @partial(jax.jit, static_argnums=0)
    def reset(self, rng: jax.Array) -> RLState:
        pipeline_state = self._static_context

        # Randomize the plant context as defined by the user-provided randomize function
        pipeline_state = self.randomize(pipeline_state, rng)

        obs = self._get_obs(pipeline_state, jnp.zeros(self._act_size))
        reward, done = jnp.zeros(2)
        metrics = {}
        return RLState(pipeline_state, obs, reward, done, metrics)  # pylint: disable=too-many-function-args

    @partial(jax.jit, static_argnums=0)
    def step(self, state: RLState, action: jax.Array) -> RLState:
        pipeline_state = self._forward(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state, action)
        reward = self.get_reward(pipeline_state, obs, action)
        done = self.get_done(pipeline_state, obs)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self) -> int:
        return self._act_size

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def backend(self) -> str:
        return "collimator"

    #
    # To be overridden by subclasses
    #
    @abc.abstractmethod
    def get_reward(
        self, pipeline_state: ContextBase, obs: jax.Array, act: jax.Array
    ) -> jax.Array:
        """Return the reward for the current state and observation."""
        pass

    def get_done(self, pipeline_state: ContextBase, obs: jax.Array) -> jax.Array:
        """Return a boolean indicating whether the episode is done."""
        return 0.0

    def randomize(self, pipeline_state: ContextBase, rng: jax.Array) -> ContextBase:
        """Randomize the initial states, parameters, etc."""
        return pipeline_state

    def render(
        self,
        trajectory: list[RLState],
        height: int = 240,
        width: int = 320,
        camera: str = None,
    ) -> list[np.ndarray]:
        """Render the trajectory"""
        raise NotImplementedError(
            "Rendering is not yet supported for Collimator environments."
        )
