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

import pytest

import jax
import jax.numpy as jnp

import collimator
from collimator import library
from collimator.optimization import RLEnv
from collimator.testing import requires_jax


def make_submodel(name="double_integrator"):
    builder = collimator.DiagramBuilder()
    int_v = builder.add(library.Integrator(initial_state=[0.0], name="int_v"))
    int_x = builder.add(library.Integrator(initial_state=[0.0], name="int_x"))

    mux = builder.add(library.Multiplexer(2, name="state"))
    builder.connect(int_x.output_ports[0], mux.input_ports[0])
    builder.connect(int_v.output_ports[0], mux.input_ports[1])

    builder.connect(int_v.output_ports[0], int_x.input_ports[0])
    builder.export_input(int_v.input_ports[0], "u")
    builder.export_output(mux.output_ports[0], "x")

    return builder.build(name=name)


class DoubleIntegratorEnv(RLEnv):
    def __init__(self, dt):
        plant = make_submodel(name="plant")
        self.int_x_id = plant["int_x"].system_id
        self.int_v_id = plant["int_v"].system_id
        super().__init__(plant, act_size=1, dt=dt)

    def randomize(self, context, key):
        x0, v0 = jax.random.uniform(key, (2, 1))
        x_context = context[self.int_x_id].with_continuous_state(x0)
        v_context = context[self.int_v_id].with_continuous_state(v0)
        return context.with_subcontext(self.int_x_id, x_context).with_subcontext(
            self.int_v_id, v_context
        )

    def get_reward(self, context, obs, act):
        x = obs
        Q = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        xQx = jnp.dot(x, jnp.dot(Q, x))
        uRu = 1e-2 * jnp.sum(jnp.square(act))
        return -self.dt * (xQx + uRu)

    def get_done(self, context, obs):
        x = context[self.int_x_id].continuous_state[0]
        v = context[self.int_v_id].continuous_state[0]
        return jnp.where(
            (jnp.abs(x) < 0.01) & (jnp.abs(v) < 0.01),
            1.0,
            0.0,
        )


@pytest.fixture
def env():
    dt = 0.1
    return DoubleIntegratorEnv(dt)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@requires_jax()
def test_initialization(env):
    assert env is not None
    assert env.action_size == 1
    assert hasattr(env, "int_x_id")
    assert hasattr(env, "int_v_id")


@requires_jax()
def test_step_function(env, rng):
    initial_state = env.reset(rng)
    action = jnp.array([0.1])
    next_state = env.step(initial_state, action)

    assert next_state is not None
    assert isinstance(next_state.obs, jax.Array)
    assert isinstance(next_state.reward, jax.Array)
    assert isinstance(next_state.done, jax.Array)


@requires_jax()
def test_reward_calculation(env, rng):
    initial_state = env.reset(rng)
    action = jnp.array([0.1])
    obs = env._get_obs(initial_state.pipeline_state, action)
    reward = env.get_reward(initial_state.pipeline_state, obs, action)

    assert reward is not None
    assert isinstance(reward, jax.Array)


@requires_jax()
def test_termination_condition(env, rng):
    initial_state = env.reset(rng)
    action = jnp.array([0.0])
    next_state = env.step(initial_state, action)

    done = env.get_done(next_state.pipeline_state, next_state.obs)

    assert done is not None
    assert isinstance(done, jax.Array)


@requires_jax()
def test_randomization(env, rng):
    context = env.reset(rng).pipeline_state
    randomized_context = env.randomize(context, rng)

    assert randomized_context is not None
    assert isinstance(randomized_context, collimator.framework.ContextBase)
