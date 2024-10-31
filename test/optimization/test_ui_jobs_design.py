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

"""
Standalone tests for optimization framework. No json processing is performed.
Incoming jobs from the UI are artificially generated.
"""

import platform
import pytest
import numpy as np

from collimator import DiagramBuilder, Parameter, SimulatorOptions
from collimator.library import (
    Adder,
    Constant,
    Gain,
    Integrator,
    Power,
    Clock,
)
from collimator.optimization import ui_jobs
from collimator.optimization.framework.base.optimizable import (
    DesignParameter,
    Distribution,
    StochasticParameter,
)


def _make_diagram(m=1.0, v0=0.1, x0=1.0):
    model_parameters = {
        "c": Parameter(np.array(1.0)),
        "k": Parameter(np.array(1.0)),
    }

    builder = DiagramBuilder()
    k_x = builder.add(Gain(model_parameters["k"], name="k_x"))
    c_v = builder.add(Gain(model_parameters["c"], name="c_v"))
    adder = builder.add(Adder(2, operators="--", name="adder"))
    one_over_m = builder.add(Gain(1.0 / m, name="one_over_m"))
    v = builder.add(Integrator(v0, name="v"))
    x = builder.add(Integrator(x0, name="x"))

    builder.connect(k_x.output_ports[0], adder.input_ports[0])
    builder.connect(c_v.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], one_over_m.input_ports[0])
    builder.connect(one_over_m.output_ports[0], v.input_ports[0])
    builder.connect(v.output_ports[0], x.input_ports[0])

    builder.connect(v.output_ports[0], c_v.input_ports[0])
    builder.connect(x.output_ports[0], k_x.input_ports[0])

    ref_x = builder.add(Constant(0.0, name="ref_x"))
    ref_v = builder.add(Constant(0.0, name="ref_v"))

    err_v = builder.add(Adder(2, operators="+-", name="err_v"))
    err_x = builder.add(Adder(2, operators="+-", name="err_x"))
    sq_err_v = builder.add(Power(2.0, name="sq_err_v"))
    sq_err_x = builder.add(Power(2.0, name="sq_err_x"))
    cost_v = builder.add(Integrator(0.0, name="cost_v"))
    cost_x = builder.add(Integrator(0.0, name="cost_x"))
    objective = builder.add(Adder(2, operators="++", name="objective"))

    builder.connect(ref_v.output_ports[0], err_v.input_ports[0])
    builder.connect(v.output_ports[0], err_v.input_ports[1])
    builder.connect(ref_x.output_ports[0], err_x.input_ports[0])
    builder.connect(x.output_ports[0], err_x.input_ports[1])

    builder.connect(err_v.output_ports[0], sq_err_v.input_ports[0])
    builder.connect(err_x.output_ports[0], sq_err_x.input_ports[0])
    builder.connect(sq_err_v.output_ports[0], cost_v.input_ports[0])
    builder.connect(sq_err_x.output_ports[0], cost_x.input_ports[0])
    builder.connect(cost_v.output_ports[0], objective.input_ports[0])
    builder.connect(cost_x.output_ports[0], objective.input_ports[1])

    # constraints
    constraint = builder.add(Adder(2, operators="+-", name="constraint"))
    min_objective_value = builder.add(Constant(1.67, name="min_objective_value"))
    builder.connect(objective.output_ports[0], constraint.input_ports[0])
    builder.connect(min_objective_value.output_ports[0], constraint.input_ports[1])

    diagram = builder.build(parameters=model_parameters)

    return (
        diagram,
        diagram["objective"].output_ports[0],
        diagram["constraint"].output_ports[0],
    )


@pytest.mark.slow
def test_optimization_unbounded():
    sim_t_span = (0.0, 2.0)

    job_type = "design"
    design_parameters = DesignParameter(
        param_name="c", initial=0.5, min=-np.inf, max=np.inf
    )

    sim_options = SimulatorOptions(max_major_steps=1)

    # Scipy with L-BFGS-B
    diagram, objective_port, _ = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="l_bfgs_b",
        options={"maxiter": 20},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        print_every=10,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 1.65, atol=0.1)

    sim_options = SimulatorOptions(max_major_steps=1)

    # Scipy Nelder-Mead
    diagram, objective_port, _ = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="nelder_mead",
        options={"maxiter": 20},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        print_every=10,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 1.65, atol=0.1)

    sim_options = SimulatorOptions(max_major_steps=1)

    # Optax sgd
    diagram, objective_port, _ = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="stochastic_gradient_descent",
        options={"learning_rate": 0.01, "num_epochs": 2000},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        print_every=200,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 1.65, atol=0.1)

    sim_options = SimulatorOptions(max_major_steps=1)

    # Evosax PSO
    diagram, objective_port, _ = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="particle_swarm_optimization",
        options={"pop_size": 50, "num_generations": 100},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        print_every=10,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 1.65, atol=0.1)


@pytest.mark.slow
def test_optimization_bounded():
    sim_t_span = (0.0, 2.0)

    job_type = "design"
    design_parameters = DesignParameter(param_name="c", initial=0.5, min=0.0, max=1.62)

    sim_options = SimulatorOptions(max_major_steps=1)

    # Scipy with L-BFGS-B
    diagram, objective_port, _ = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="l_bfgs_b",
        options={"maxiter": 20},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        print_every=10,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 1.62, atol=0.1)

    sim_options = SimulatorOptions(max_major_steps=1)

    # Evosax with PSO
    diagram, objective_port, _ = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="particle_swarm_optimization",
        options={"pop_size": 50, "num_generations": 100},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        print_every=10,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 1.62, atol=0.01)


@pytest.mark.slow
def test_optimization_constrained_scipy():
    sim_t_span = (0.0, 2.0)

    job_type = "design"
    design_parameters = DesignParameter(param_name="c", initial=0.5, min=0.0, max=2.0)

    sim_options = SimulatorOptions(max_major_steps=1)

    # Scipy SLSQP
    diagram, objective_port, constraint_port = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="sequential_least_squares",
        options={"maxiter": 20},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        constraint_ports=[constraint_port],
        print_every=10,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 0.36, atol=0.1)


@pytest.mark.slow
@pytest.mark.skipif(
    platform.machine() == "arm64", reason="nlopt not available for arm64"
)
def test_optimization_constrained_nlopt():
    sim_t_span = (0.0, 2.0)

    job_type = "design"
    design_parameters = DesignParameter(param_name="c", initial=0.5, min=0.0, max=2.0)

    sim_options = SimulatorOptions(max_major_steps=1)

    # NLopt Direct
    diagram, objective_port, constraint_port = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="dividing_rectangles",
        options={},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        constraint_ports=[constraint_port],
        print_every=10,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 0.36, atol=0.1)


@pytest.mark.slow
def test_optimization_stochastic():
    sim_t_span = (0.0, 2.0)

    job_type = "design"
    design_parameters = DesignParameter(
        param_name="c", initial=0.5, min=-np.inf, max=np.inf
    )

    stochastic_var = StochasticParameter(
        param_name="k",
        distribution=Distribution(name="uniform", options={"min": 0.5, "max": 1.5}),
    )

    sim_options = SimulatorOptions(max_major_steps=1)

    # Optax SGD
    diagram, objective_port, _ = _make_diagram()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="stochastic_gradient_descent",
        options={"learning_rate": 0.1, "num_epochs": 50},
        design_parameters=[design_parameters],
        stochastic_parameters=[stochastic_var],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        print_every=1,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")

    np.testing.assert_allclose(opt_param["c"], 1.74, atol=0.2)


@pytest.mark.slow
def test_optimization_initial_state():
    """Test for optimizing model parameters used as an initial_state."""
    sim_t_span = (0.0, 2.0)

    def _make_diagram_with_init_state():
        """Clock connected to a double integrator with initial states set to c.
        The objective is to minimize the squared error between the integrator
        output and a constant 120.0. Meaning that the final value of the integrator
        should be close to 120.0.
        """
        builder = DiagramBuilder()
        c = Parameter(0.0)
        Constant_0 = Constant(120.0)
        builder.add(Constant_0)
        Adder_0 = Adder(2, operators="+-")
        builder.add(Adder_0)
        Error = Power(2.0)
        builder.add(Error)
        Integrator_0 = Integrator(
            c, lower_limit=-1.0, upper_limit=1.0, enable_external_reset=True
        )
        builder.add(Integrator_0)
        Integrator_1 = Integrator(
            c, lower_limit=-1.0, upper_limit=1.0, enable_external_reset=True
        )
        builder.add(Integrator_1)
        Clock_0 = Clock()
        builder.add(Clock_0)
        builder.connect(Integrator_0.output_ports[0], Adder_0.input_ports[1])
        builder.connect(Constant_0.output_ports[0], Adder_0.input_ports[0])
        builder.connect(Adder_0.output_ports[0], Error.input_ports[0])
        builder.connect(Integrator_1.output_ports[0], Integrator_0.input_ports[0])
        builder.connect(Clock_0.output_ports[0], Integrator_1.input_ports[0])
        return builder.build("root", parameters={"c": c}), Error.output_ports[0]

    job_type = "design"
    design_parameters = DesignParameter(
        param_name="c", initial=0.0, min=-np.inf, max=np.inf
    )

    sim_options = SimulatorOptions(max_major_steps=1)

    diagram, objective_port = _make_diagram_with_init_state()
    opt_param, _ = ui_jobs.jobs_router(
        job_type,
        diagram,
        algorithm="stochastic_gradient_descent",
        options={"learning_rate": 0.1, "num_epochs": 2000},
        design_parameters=[design_parameters],
        sim_t_span=sim_t_span,
        objective_port=objective_port,
        print_every=200,
        sim_options=sim_options,
    )
    print(f"{opt_param=}")
    np.testing.assert_allclose(opt_param["c"], 39.5, atol=0.1)


if __name__ == "__main__":
    # test_optimization_unbounded()
    # test_optimization_bounded()
    # test_optimization_constrained_scipy()
    # test_optimization_constrained_nlopt()
    test_optimization_stochastic()
