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

import json
import numpy as np
import pytest

from collimator import DiagramBuilder, LeafSystem, Simulator, SimulatorOptions, simulate
from collimator.dashboard.serialization import to_model_json
from collimator.framework import parameters, Parameter
from collimator.library import (
    Constant,
    Gain,
    Abs,
    Integrator,
    Adder,
    PID,
    TransferFunction,
)
from collimator.simulation.types import SimulatorState
from collimator.testing import set_backend
from collimator.backend import numpy_api as cnp


class GainWithStaticParam(LeafSystem):
    @parameters(static=["p"])
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_input_port()
        self._output_port_idx = self.declare_output_port(
            None,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )

    def initialize(self, p):
        # This invalidates the compute graph and 'recorded_signals' and requires
        # a recompilation (for JAX).
        self.configure_output_port(
            self._output_port_idx,
            lambda time, state, u: p * u,
            prerequisites_of_calc=[self.input_ports[0].ticket],
            requires_inputs=True,
        )


# Simulate will ALWAYS recompile the model (at least given current implementation
# that involves re-creating sim and thus sim.advance_to). So, this test passes.
@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_update_static_param_simulate(backend, mode):
    set_backend(backend)

    p = Parameter(1.0)
    builder = DiagramBuilder()
    constant = builder.add(Constant(1.0))
    gain = builder.add(Gain(p) if mode == "dynamic" else GainWithStaticParam(p))
    builder.connect(constant.output_ports[0], gain.input_ports[0])
    diagram = builder.build()

    context = diagram.create_context()
    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "output": gain.output_ports[0],
        },
    )
    assert results.outputs["output"][0] == 1.0

    p.set(2.0)
    context = diagram.create_context()
    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "output": gain.output_ports[0],
        },
    )
    assert results.outputs["output"][0] == 2.0


# Now, do the same but with JIT'ed advance_to like ensemble sims.
# This would not recompile the model, and we expect this test to fail with JAX, but
# kinda work with Numpy.
@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_update_static_param_no_compilation(backend, mode):
    set_backend(backend)

    # Enable this to observe JAX cache misses. It can help understanding why JAX
    # recompiles (or not).
    # jax.config.update("jax_explain_cache_misses", True)

    p = Parameter(0.0)
    builder = DiagramBuilder()
    constant = builder.add(Constant(1.0))
    gain = builder.add(Gain(p) if mode == "dynamic" else GainWithStaticParam(p))
    builder.connect(constant.output_ports[0], gain.input_ports[0])
    diagram = builder.build()

    recorded_signals = {"output": gain.output_ports[0]}
    options = SimulatorOptions(recorded_signals=recorded_signals, save_time_series=True)
    simulator = Simulator(diagram, options=options)
    advance_to = cnp.jit(simulator.advance_to)

    for gain_value in range(10):
        p.set(float(gain_value))
        context = diagram.create_context(time=0.0, check_types=False)

        print('id(recorded_signals["output"])', id(recorded_signals["output"]))
        print("id(gain.output_ports[0])      ", id(gain.output_ports[0]))
        if id(recorded_signals["output"]) != id(gain.output_ports[0]):
            print('WARNING: We observed that recorded_signals["output"] changed')
            recorded_signals["output"] = gain.output_ports[0]

        simstate: SimulatorState = advance_to(1.0, context)
        _times, outputs = simstate.results_data.finalize()

        if mode == "static":
            assert len(p.static_dependents) == 1

        try:
            np.testing.assert_allclose(outputs["output"][0], float(gain_value))
        except AssertionError as e:
            print("Got an error:", e)
            print(f"Backend: {backend}, Mode: {mode}")
            print(f"Expected: {gain_value}, Got: {outputs['output'][0]}")
            if mode == "static" and backend == "jax":
                # Basically, this should be an error upstream, we should not be able to
                # reach this point. The ensemble sim runner will appropriately fail if
                # it detects static dependents, but there will be cases where we fail
                # to detect those.
                pytest.xfail("This test highlights a bug with JAX + static parameters")
            raise e


# Failing test case for WC-462
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_invalid_pid_ensemble_sim(backend):
    set_backend(backend)

    kp = Parameter(1.0, name="kp")
    builder = DiagramBuilder()
    ref = builder.add(Constant(1.0))
    err = builder.add(Adder(n_in=2, operators="+-"))
    pid = builder.add(PID(kp=kp, ki=0, kd=0, n=100))
    tf = builder.add(TransferFunction([1], [1, 1]))
    err_abs = builder.add(Abs())
    err_cum = builder.add(Integrator(initial_state=0.0))
    builder.connect(ref.output_ports[0], err.input_ports[0])
    builder.connect(tf.output_ports[0], err.input_ports[1])
    builder.connect(err.output_ports[0], pid.input_ports[0])
    builder.connect(pid.output_ports[0], tf.input_ports[0])
    builder.connect(err.output_ports[0], err_abs.input_ports[0])
    builder.connect(err_abs.output_ports[0], err_cum.input_ports[0])
    diagram = builder.build(
        parameters={"kp": kp},
    )
    diagram.pprint()

    model, _ = to_model_json.convert(diagram)
    print(json.dumps(model.to_api()))

    recorded_signals = {"error": err_cum.output_ports[0]}
    options = SimulatorOptions(recorded_signals=recorded_signals, save_time_series=True)
    simulator = Simulator(diagram, options=options)
    advance_to = cnp.jit(simulator.advance_to)

    # Set Kp to 0.0 - makes the block non-feedthrough
    kp.set(0.0)
    context = diagram.create_context(time=0.0, check_types=True)
    recorded_signals["output"] = err_cum.output_ports[0]
    simstate: SimulatorState = advance_to(1.0, context)
    times_kp_0, outputs = simstate.results_data.finalize()
    results_kp_0 = np.array(outputs["error"], copy=True)
    print("results_kp_0", results_kp_0)

    # Now change to 1.0 - makes the block feedthrough
    # This requires a reconfiguration & recompilation, which will fail with JAX
    kp.set(1.0)
    context = diagram.create_context(time=0.0, check_types=True)
    recorded_signals["output"] = err_cum.output_ports[0]
    simstate: SimulatorState = advance_to(1.0, context)
    times_kp_1, outputs = simstate.results_data.finalize()
    results_kp_1 = np.array(outputs["error"], copy=True)
    print("results_kp_1", results_kp_1)

    # FIXME this does not belong here
    def resample(x, y, x_new):
        results_resampled = np.zeros_like(x_new)
        for i in range(len(x_new)):
            results_resampled[i] = np.interp(x_new[i], x, y, left=y[0])
        return results_resampled

    results_kp_1_resampled = resample(times_kp_1, results_kp_1, times_kp_0)
    print("results_kp_1_resampled", results_kp_1_resampled)

    if backend == "numpy":
        assert not np.allclose(results_kp_0, results_kp_1_resampled)

    if backend == "jax":
        # THIS CHECK IS INVALID! THIS IS TESTING A BUG!
        # If this bug finally gets fixed, amend this check accordingly.
        np.testing.assert_allclose(results_kp_0, results_kp_1)
        pytest.xfail(
            "The results are the same after changing Kp! "
            "They should have changed! This tests a known bug."
        )


if __name__ == "__main__":
    # test_update_static_param("numpy")
    # test_update_static_param_simulate("jax", "static")
    # test_update_static_param_no_compilation("jax", "static")
    test_invalid_pid_ensemble_sim("numpy")
    test_invalid_pid_ensemble_sim("jax")
