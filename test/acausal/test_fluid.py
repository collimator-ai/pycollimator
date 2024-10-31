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

import numpy as np
from matplotlib import pyplot as plt
import pytest
import sympy as sp

# acausal imports
from collimator.experimental import AcausalCompiler, AcausalDiagram, EqnEnv
from collimator.experimental import fluid as fld
from collimator.experimental import fluid_media as fm
from collimator.experimental import thermal
from collimator.experimental.acausal.component_library.fluid import P_DEFAULT, T_DEFAULT


# collimator imports
import collimator

import collimator.logging as logging
from collimator.testing.markers import skip_if_not_jax

logging.set_log_level(logging.DEBUG)
skip_if_not_jax()

###########################################################
# Tests with Fluid IdealGasAir
###########################################################


@pytest.mark.flaky(retries=3)
@pytest.mark.parametrize("tank_pipe_tank", [True, False])
def test_tank_pipe_amb(tank_pipe_tank, show_plot=False, use_ida=False):
    """
    The simplest model possible.
    tank<->pipe<->boundary_conditions.
    if the tank had a MassFlowSource connected to it, this mode would be identical
    to the OpenModelica PartialTestModel in the Media library.

    optionally run as tank-pipe-tank, as this has similar dynamics, but requires different
    outcome from diagram processing.
    """
    p_ic = 101325.0
    ev = EqnEnv()
    fp = fld.FluidProperties(ev, fluid=fm.IdealGasAir(ev))
    ad = AcausalDiagram()
    cv = fld.ClosedVolume(ev, name="cv1", pressure_ic=p_ic + 100)
    pipe = fld.SimplePipe(ev, name="pipe", R=100.0)
    if tank_pipe_tank:
        amb = fld.ClosedVolume(ev, name="cv2", pressure_ic=p_ic)
    else:
        amb = fld.Boundary_pT(ev, name="amb", p_ambient=p_ic)
    ad.connect(cv, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", amb, "port")

    # the FluidProp comp is connected to the ClosedVolume, because the FluidProp
    # data is needed by all components, Boundary_pT included, so this indirect connection between
    # the FluidProp comp and the Boundary_pT tests whether DiagramProcessing can correctly
    # assign the FluidProps for the Boundary_pT.
    ad.connect(fp, "prop", cv, "port")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, scale=True, verbose=True)

    t_end = 10.0
    if use_ida:
        acausal_system, sed = ac(return_sed=True)
        from collimator.experimental.acausal.index_reduction.ida_solver import (
            IDASolver,
        )

        idasolver = IDASolver(sed)
        t, x = idasolver.solve(t_end, dt=0.001, exclude_algvar_from_error=False)
    else:
        # FIXME: simulation fails because step size too small error.
        acausal_system = ac()
        builder = collimator.DiagramBuilder()
        acausal_system = builder.add(acausal_system)
        diagram = builder.build()
        context = diagram.create_context(check_types=True)
        recorded_signals = {
            "x": acausal_system.output_ports[0],
        }
        results = collimator.simulate(
            diagram,
            context,
            (0.0, t_end),
            recorded_signals=recorded_signals,
        )
        t = results.time
        x = results.outputs["x"]
    if show_plot:
        nstates = x.shape[1]
        fig, axs = plt.subplots(nstates, 1, figsize=(8, nstates))
        for i, ax in enumerate(axs):
            label = "x" + str(i)
            ax.plot(t, x[:, i], label=label)
            ax.legend()
        fig.tight_layout()
        plt.show()


def test_tank_pipe_amb_with_sensors(show_plot=False, use_ida=False):
    # CV-|-sensPT
    #    |-pipe1-sensM-pipe2-amb
    # NOTE: presently the sensor is meansuring pressure and enthalpy.
    p_ic = 101325.0
    ev = EqnEnv()
    fp = fld.FluidProperties(ev, fluid=fm.IdealGasAir(ev))
    ad = AcausalDiagram()
    cv = fld.ClosedVolume(ev, name="CV", pressure_ic=p_ic + 100)
    pipe1 = fld.SimplePipe(ev, name="pipe1", R=5000.0, enable_sensors=True)
    sensM = fld.MassflowSensor(ev, name="sensM")
    pipe2 = fld.SimplePipe(ev, name="pipe2", R=5000.0, enable_sensors=True)
    amb = fld.Boundary_pT(ev, name="amb", p_ambient=p_ic)
    sensPT = fld.PTSensor(ev, name="sensPT", enable_port_b=False)
    ad.connect(cv, "port", pipe1, "port_a")
    ad.connect(pipe1, "port_b", sensM, "port_a")
    ad.connect(sensM, "port_b", pipe2, "port_a")
    ad.connect(pipe2, "port_b", amb, "port")
    ad.connect(fp, "prop", cv, "port")
    ad.connect(cv, "port", sensPT, "port_a")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, scale=False, verbose=True)

    t_end = 1.0
    if use_ida:
        acausal_system, sed = ac(return_sed=True)
        from collimator.experimental.acausal.index_reduction.ida_solver import (
            IDASolver,
        )

        idasolver = IDASolver(sed)
        t, x = idasolver.solve(t_end, dt=0.001, exclude_algvar_from_error=False)
    else:
        acausal_system = ac()
        builder = collimator.DiagramBuilder()
        acausal_system = builder.add(acausal_system)
        diagram = builder.build()
        context = diagram.create_context(check_types=True)

        sensM_idx = acausal_system.outsym_to_portid[
            sensM.get_sym_by_port_name("m_flow")
        ]
        pipe1M_idx = acausal_system.outsym_to_portid[
            pipe1.get_sym_by_port_name("m_flow")
        ]
        pipe2M_idx = acausal_system.outsym_to_portid[
            pipe2.get_sym_by_port_name("m_flow")
        ]
        pipe1Pa_idx = acausal_system.outsym_to_portid[pipe1.get_sym_by_port_name("pa")]
        pipe2Pa_idx = acausal_system.outsym_to_portid[pipe2.get_sym_by_port_name("pa")]
        pipe1Pb_idx = acausal_system.outsym_to_portid[pipe1.get_sym_by_port_name("pb")]
        pipe2Pb_idx = acausal_system.outsym_to_portid[pipe2.get_sym_by_port_name("pb")]
        sensP_idx = acausal_system.outsym_to_portid[sensPT.get_sym_by_port_name("p")]
        sensT_idx = acausal_system.outsym_to_portid[sensPT.get_sym_by_port_name("temp")]
        recorded_signals = {
            "sensM": acausal_system.output_ports[sensM_idx],
            "pipe1M": acausal_system.output_ports[pipe1M_idx],
            "pipe2M": acausal_system.output_ports[pipe2M_idx],
            "pipe1Pa": acausal_system.output_ports[pipe1Pa_idx],
            "pipe2Pa": acausal_system.output_ports[pipe2Pa_idx],
            "pipe1Pb": acausal_system.output_ports[pipe1Pb_idx],
            "pipe2Pb": acausal_system.output_ports[pipe2Pb_idx],
            "sensP": acausal_system.output_ports[sensP_idx],
            "sensT": acausal_system.output_ports[sensT_idx],
        }

        results = collimator.simulate(
            diagram,
            context,
            (0.0, t_end),
            recorded_signals=recorded_signals,
        )
        t = results.time
        sensM = results.outputs["sensM"]
        pipe1M = results.outputs["pipe1M"]
        pipe2M = results.outputs["pipe2M"]
        pipe1Pa = results.outputs["pipe1Pa"]
        pipe2Pa = results.outputs["pipe2Pa"]
        pipe1Pb = results.outputs["pipe1Pb"]
        pipe2Pb = results.outputs["pipe2Pb"]
        sensP = results.outputs["sensP"]
        sensT = results.outputs["sensT"]

        sensM_sol = sensM[0] * (np.exp(-t / (0.099)))

        if show_plot:
            fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))
            ax.plot(t, sensM, label="sensM", linewidth=5)
            ax.plot(t, pipe1M, label="pipe1M", linewidth=3)
            ax.plot(t, pipe2M, label="pipe2M")
            ax.legend()

            ax2.plot(t, pipe1Pa, label="pipe1Pa", linewidth=3)
            ax2.plot(t, pipe2Pa, label="pipe2Pa", linewidth=3)
            ax2.plot(t, pipe1Pb, label="pipe1Pb")
            ax2.plot(t, pipe2Pb, label="pipe2Pb")
            ax2.plot(t, sensP, label="sensP")
            ax2.legend()

            ax3.plot(t, sensT, label="sensPT h_inStream")
            ax3.legend()

            ax4.plot(t, sensM, label="sensM", linewidth=5)
            ax4.plot(t, sensM_sol, label="sensM_sol")
            ax4.legend()
            plt.show()

        atol = 0.01
        rtol = 0.01
        assert np.allclose(sensM_sol, sensM, atol=atol, rtol=rtol)


def test_T_junction_splitting(show_plot=False):
    """
    The simplest model that can be made for testing an ideal splitting
    flow scenario. It's 'splitting' because the pressures will result in Amb
    flowing to both CV1 and CV2.
                ,-pipe1-CV1
    Amb-pipeAmb-|
                `-pipe2-CV2
    """
    p_ic = 101325.0
    ev = EqnEnv()
    fp = fld.FluidProperties(ev, fluid=fm.IdealGasAir(ev))
    ad = AcausalDiagram()
    cv1 = fld.ClosedVolume(ev, name="cv1", enable_enthalpy_sensor=True)
    cv2 = fld.ClosedVolume(ev, name="cv2", enable_enthalpy_sensor=True)
    pipe1 = fld.SimplePipe(ev, name="pipe1", R=20000.0, enable_sensors=True)
    pipe2 = fld.SimplePipe(ev, name="pipe2", R=10000.0, enable_sensors=True)
    pipeAmb = fld.SimplePipe(ev, name="pipeAmb")
    amb = fld.Boundary_pT(ev, name="amb", p_ambient=p_ic + 100)
    ad.connect(cv1, "port", pipe1, "port_a")
    ad.connect(cv2, "port", pipe2, "port_a")
    ad.connect(amb, "port", pipeAmb, "port_a")
    ad.connect(pipeAmb, "port_b", pipe1, "port_b")
    ad.connect(pipeAmb, "port_b", pipe2, "port_b")
    ad.connect(fp, "prop", amb, "port")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()

    t_end = 1.0
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # results capture
    pipe1M_idx = acausal_system.outsym_to_portid[pipe1.get_sym_by_port_name("m_flow")]
    pipe2M_idx = acausal_system.outsym_to_portid[pipe2.get_sym_by_port_name("m_flow")]
    pipe1Pa_idx = acausal_system.outsym_to_portid[pipe1.get_sym_by_port_name("pa")]
    pipe2Pa_idx = acausal_system.outsym_to_portid[pipe2.get_sym_by_port_name("pa")]
    pipe1Pb_idx = acausal_system.outsym_to_portid[pipe1.get_sym_by_port_name("pb")]
    pipe2Pb_idx = acausal_system.outsym_to_portid[pipe2.get_sym_by_port_name("pb")]
    cv1h_idx = acausal_system.outsym_to_portid[cv1.get_sym_by_port_name("h_output")]
    cv2h_idx = acausal_system.outsym_to_portid[cv2.get_sym_by_port_name("h_output")]
    recorded_signals = {
        "pipe1M": acausal_system.output_ports[pipe1M_idx],
        "pipe2M": acausal_system.output_ports[pipe2M_idx],
        "pipe1Pa": acausal_system.output_ports[pipe1Pa_idx],
        "pipe2Pa": acausal_system.output_ports[pipe2Pa_idx],
        "pipe1Pb": acausal_system.output_ports[pipe1Pb_idx],
        "pipe2Pb": acausal_system.output_ports[pipe2Pb_idx],
        "cv1h": acausal_system.output_ports[cv1h_idx],
        "cv2h": acausal_system.output_ports[cv2h_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals=recorded_signals,
    )
    t = results.time
    pipe1M = results.outputs["pipe1M"]
    pipe2M = results.outputs["pipe2M"]
    pipe1Pa = results.outputs["pipe1Pa"]
    pipe2Pa = results.outputs["pipe2Pa"]
    pipe1Pb = results.outputs["pipe1Pb"]
    pipe2Pb = results.outputs["pipe2Pb"]
    cv1h = results.outputs["cv1h"]
    cv2h = results.outputs["cv2h"]

    if show_plot:
        fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
        ax.plot(t, pipe1M, label="pipe1M", linewidth=3)
        ax.plot(t, pipe2M, label="pipe2M")
        ax.legend()

        ax2.plot(t, pipe1Pa, label="pipe1Pa", linewidth=3)
        ax2.plot(t, pipe2Pa, label="pipe2Pa")
        ax2.plot(t, pipe1Pb, label="pipe1Pb", linewidth=3)
        ax2.plot(t, pipe2Pb, label="pipe2Pb")
        ax2.legend()

        ax3.plot(t, cv1h, label="cv1h", linewidth=3)
        ax3.plot(t, cv2h, label="cv2h")
        ax3.legend()
        plt.show()


def test_T_junction_merging(show_plot=False):
    """
    The simplest model that can be made for testing an ideal merging
    flow scenario. It's 'merging' because the pressures will result in CV1
    and CV2 flowing into CV0.
                ,-Bpipe1A-CV1
    CV0-Apipe0B-|
                `-Bpipe2A-CV2
    """
    p_ic = 101325.0
    ev = EqnEnv()
    fp = fld.FluidProperties(ev, fluid=fm.IdealGasAir(ev))
    ad = AcausalDiagram()
    cv0 = fld.ClosedVolume(
        ev,
        name="cv0",
        pressure_ic=p_ic,
        enable_enthalpy_sensor=True,
    )
    cv1 = fld.ClosedVolume(
        ev,
        name="cv1",
        pressure_ic=p_ic + 300,
        enable_enthalpy_sensor=True,
    )
    cv2 = fld.ClosedVolume(
        ev,
        name="cv2",
        pressure_ic=p_ic + 100,
        enable_enthalpy_sensor=True,
    )
    pipe0 = fld.SimplePipe(ev, name="pipe0", R=50000.0, enable_sensors=True)
    pipe1 = fld.SimplePipe(ev, name="pipe1", R=10000.0, enable_sensors=True)
    pipe2 = fld.SimplePipe(ev, name="pipe2", R=10000.0, enable_sensors=True)
    ad.connect(cv1, "port", pipe1, "port_a")
    ad.connect(cv2, "port", pipe2, "port_a")
    ad.connect(cv0, "port", pipe0, "port_a")
    ad.connect(pipe0, "port_b", pipe1, "port_b")
    ad.connect(pipe0, "port_b", pipe2, "port_b")
    ad.connect(fp, "prop", cv0, "port")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()

    t_end = 1.0
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)

    # results capture
    pipe0M_idx = acausal_system.outsym_to_portid[pipe0.get_sym_by_port_name("m_flow")]
    pipe1M_idx = acausal_system.outsym_to_portid[pipe1.get_sym_by_port_name("m_flow")]
    pipe2M_idx = acausal_system.outsym_to_portid[pipe2.get_sym_by_port_name("m_flow")]
    pipe0Pa_idx = acausal_system.outsym_to_portid[pipe0.get_sym_by_port_name("pa")]
    pipe1Pa_idx = acausal_system.outsym_to_portid[pipe1.get_sym_by_port_name("pa")]
    pipe2Pa_idx = acausal_system.outsym_to_portid[pipe2.get_sym_by_port_name("pa")]
    pipe0Pb_idx = acausal_system.outsym_to_portid[pipe0.get_sym_by_port_name("pb")]
    cv0h_idx = acausal_system.outsym_to_portid[cv0.get_sym_by_port_name("h_output")]
    cv1h_idx = acausal_system.outsym_to_portid[cv1.get_sym_by_port_name("h_output")]
    cv2h_idx = acausal_system.outsym_to_portid[cv2.get_sym_by_port_name("h_output")]
    recorded_signals = {
        "pipe0M": acausal_system.output_ports[pipe0M_idx],
        "pipe1M": acausal_system.output_ports[pipe1M_idx],
        "pipe2M": acausal_system.output_ports[pipe2M_idx],
        "pipe0Pa": acausal_system.output_ports[pipe0Pa_idx],
        "pipe1Pa": acausal_system.output_ports[pipe1Pa_idx],
        "pipe2Pa": acausal_system.output_ports[pipe2Pa_idx],
        "pipe0Pb": acausal_system.output_ports[pipe0Pb_idx],
        "cv0h": acausal_system.output_ports[cv0h_idx],
        "cv1h": acausal_system.output_ports[cv1h_idx],
        "cv2h": acausal_system.output_ports[cv2h_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals=recorded_signals,
    )
    t = results.time
    pipe0M = results.outputs["pipe0M"]
    pipe1M = results.outputs["pipe1M"]
    pipe2M = results.outputs["pipe2M"]
    pipe0Pa = results.outputs["pipe0Pa"]
    pipe1Pa = results.outputs["pipe1Pa"]
    pipe2Pa = results.outputs["pipe2Pa"]
    pipe0Pb = results.outputs["pipe0Pb"]
    cv0h = results.outputs["cv0h"]
    cv1h = results.outputs["cv1h"]
    cv2h = results.outputs["cv2h"]

    if show_plot:
        fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
        ax.plot(t, pipe0M, label="pipe0M", linewidth=5)
        ax.plot(t, pipe1M, label="pipe1M", linewidth=3)
        ax.plot(t, pipe2M, label="pipe2M")
        ax.grid()
        ax.legend()

        ax2.plot(t, pipe0Pa, label="pipe0Pa", linewidth=5)
        ax2.plot(t, pipe1Pa, label="pipe1Pa", linewidth=3)
        ax2.plot(t, pipe2Pa, label="pipe2Pa")
        ax2.plot(t, pipe0Pb, label="pipe0Pb")
        ax2.grid()
        ax2.legend()

        ax3.plot(t, cv0h, label="cv0h", linewidth=5)
        ax3.plot(t, cv1h, label="cv1h", linewidth=3)
        ax3.plot(t, cv2h, label="cv2h")
        ax3.legend()
        ax3.grid()
        plt.show()


# Cross domain tests fluid(air)<->thermal
@pytest.mark.skip(reason="might be flakey")
@pytest.mark.parametrize("use_heat_source", [True, False])
def test_heat_tank_pipe_amb(use_heat_source, show_plot=False):
    """
    A simple model demonstrating heat exchange into/out of fluid.

    heat_capacitor<->tank<->pipe<->amb.

    Initially the tank pressure is greater than ambient, which results
    in mass flow from tank to ambient. when the pressure different gets
    to zero, the heat being removed from the air in the tank by the
    thermal connection dominates, and flow reverses, going from ambient
    to the tank.
    """
    # build acausal model
    p_ic = P_DEFAULT
    fluid_T_ic = T_DEFAULT + 100  # deg K
    chiller_T_ic = T_DEFAULT  # deg K
    ev = EqnEnv()
    fp = fld.FluidProperties(
        ev,
        fluid=fm.IdealGasAir(
            ev,
            P_ic=p_ic,
            T_ic=fluid_T_ic,
        ),
    )
    ad = AcausalDiagram()
    if use_heat_source:
        heat = thermal.HeatflowSource(
            ev,
            name="hs",
            Q_flow=1,  # small power because CV has small mass.
            enable_port_b=False,
        )
    else:
        heat = thermal.HeatCapacitor(
            ev,
            name="hc",
            initial_temperature=chiller_T_ic,
            initial_temperature_fixed=True,
        )
    sensT = thermal.TemperatureSensor(ev, name="sensT", enable_port_b=False)
    cv = fld.ClosedVolume(
        ev,
        name="cv1",
        pressure_ic=p_ic + 100,
        temperature_ic=fluid_T_ic,
        temperature_ic_fixed=True,
        enable_thermal_port=True,
        enable_enthalpy_sensor=True,
    )
    pipe = fld.SimplePipe(
        ev,
        name="pipe",
        R=100000.0,
        enable_sensors=True,
    )
    amb = fld.Boundary_pT(ev, name="amb", p_ambient=p_ic, T_ambient=fluid_T_ic)
    if use_heat_source:
        ad.connect(heat, "port_a", cv, "wall")
    else:
        ad.connect(heat, "port", cv, "wall")
    ad.connect(sensT, "port_a", cv, "wall")
    ad.connect(cv, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", amb, "port")
    ad.connect(fp, "prop", cv, "port")
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()
    # run sim
    t_end = 10.0
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    pipeM_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("m_flow")]
    pipePa_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("pa")]
    pipePb_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("pb")]
    cvh_idx = acausal_system.outsym_to_portid[cv.get_sym_by_port_name("h_output")]
    sensT_idx = acausal_system.outsym_to_portid[sensT.get_sym_by_port_name("T_rel")]
    recorded_signals = {
        "pipeM": acausal_system.output_ports[pipeM_idx],
        "pipePa": acausal_system.output_ports[pipePa_idx],
        "pipePb": acausal_system.output_ports[pipePb_idx],
        "cvh": acausal_system.output_ports[cvh_idx],
        "sensT": acausal_system.output_ports[sensT_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals=recorded_signals,
    )
    # plot results
    t = results.time
    pipeM = results.outputs["pipeM"]
    pipePa = results.outputs["pipePa"]
    pipePb = results.outputs["pipePb"]
    cvh = results.outputs["cvh"]
    sensT = results.outputs["sensT"]
    if show_plot:
        fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))
        ax.plot(t, pipeM, label="pipeM")
        ax.legend()
        ax.grid()

        ax2.plot(t, pipePa, label="pipePa")
        ax2.plot(t, pipePb, label="pipePb")
        ax2.legend()
        ax2.grid()

        ax3.plot(t, cvh, label="CV h_outflow")
        ax3.legend()
        ax3.grid()

        ax4.plot(t, sensT, label="sensT")
        ax4.legend()
        ax4.grid()
        plt.show()


###########################################################
# Tests with Fluid WaterLiquidSimple or WaterLiquid
###########################################################


@pytest.mark.skip(reason="might be flakey")
def test_water_if97_region1():
    """
    Perform the 'computer-program verification' from section 5.1 of
    http://www.iapws.org/relguide/IF97-Rev.pdf
    """
    ev = EqnEnv()
    fluid = fm.WaterLiquid(ev)
    p = sp.Symbol("p")
    T = sp.Symbol("T")
    h = sp.Symbol("h")
    u = sp.Symbol("u")
    d = sp.Symbol("d")

    # values from table 5. "computer program verification values"
    # h and u values have been increased by 1e3 to change the units from
    # kJ/kg to J/kg
    T_vals = [300, 300, 500]
    p_vals = [3e6, 80e6, 3e6]
    sv_vals = [0.100215168e-2, 0.971180894e-3, 0.120241800e-2]
    h_vals = [0.115331273e6, 0.184142828e6, 0.975542239e6]
    u_vals = [0.112324818e6, 0.106448356e6, 0.971934985e6]

    eqs, gp, gt = fluid.gen_eqs(ev, p, h, T, u, d, return_gamma=True)

    gamma_args = [p, T]
    gamma_exprs = [e.rhs for e in eqs[:2]]
    # print(f"{gamma_exprs=}")
    fgamma = sp.lambdify(
        gamma_args,
        gamma_exprs,
    )

    prop_args = [ev.t, p, T, gp, gt]
    print(f"{prop_args=}")
    prop_exprs = [e.rhs for e in eqs[2:]]
    print(f"{prop_exprs=}")
    fprop = sp.lambdify(
        prop_args,
        prop_exprs,
    )

    for T_v, p_v, sv, h_sol, u_sol in zip(T_vals, p_vals, sv_vals, h_vals, u_vals):
        d_sol = 1 / sv
        # compute values from fluid equations
        print(f"\n{T_v=} {p_v=}")
        gp_v, gt_v = fgamma(p_v, T_v)
        print(f"\t{gp_v=} {gt_v=}")
        d, u, h = fprop(0.0, p_v, T_v, gp_v, gt_v)
        print(f"\t{d} {d_sol=} {d-d_sol=}")
        print(f"\t{u} {u_sol=} {u-u_sol=}")
        print(f"\t{h} {h_sol=} {h-h_sol=}")

        # compute values form IC function
        h_ic, u_ic, d_ic = fluid.get_h_u_d_ics(p_v, T_v)
        print(f"\t{d_ic=} {d_sol=} {d_ic-d_sol=}")
        print(f"\t{u_ic=} {u_sol=} {u_ic-u_sol=}")
        print(f"\t{h_ic} {h_sol=} {h_ic-h_sol=}")

        # check solutions
        sol = np.array([d_sol, u_sol, h_sol])
        eqs_vals = np.array([d, u, h])
        ics_vals = np.array([d_ic, u_ic, h_ic])
        assert np.allclose(sol, eqs_vals)
        assert np.allclose(sol, ics_vals)


@pytest.mark.parametrize("compX", ["amb", "acc", "open_tank"])
def test_acc_pipe_compX(compX, show_plot=False, use_simple_water=True):
    """
    The simplest model possible for incompressible water flow.
    accumulator<->pipe<->boundary_conditions.

    Is also re-used to make other test with same system layout:
    accumulator<->pipe<->accumulator
    accumulator<->pipe<->open_tank
    """
    p_ic = 101325.0
    fluid_T_ic = T_DEFAULT + 100 * 0  # deg K
    ev = EqnEnv()
    if use_simple_water:
        # NOTE: this fluid model has a peculiar behavior in that when the mass flows
        # out of the Accumulator, the fluid temperature increases slightly. it's an
        # insignificant increase, but still not intuitive.
        # after much debigging, I have confirmed that this is NOT due to:
        #   - the "leakage" of h_inStream flow in to the Accumulator despite mflow < 0
        #   - or energy coming from anywhere else.
        # Infact, if the pressure difference between Accumulator and Ambient is made negative,
        # thus mass flowing into the Accumulator, the temperature decreases slightly, so opposite
        # effect. At least it is consistent.
        # this could be due to the over simplification of the Fluid Media model, and the rounding errors
        # in pretty all the constants used in the model.
        fp = fld.FluidProperties(ev, fluid=fm.WaterLiquidSimple(ev))
    else:
        # FIXME: using the IF97 water model results in very long running Acaual System compilation.
        # DP.remove_duplicate_eqs and IR.BLT have been seen to be culprits.
        fp = fld.FluidProperties(ev, fluid=fm.WaterLiquid(ev))
    ad = AcausalDiagram()
    accum1 = fld.Accumulator(
        ev,
        name="accum1",
        P_ic=p_ic + 100,
        P_ic_fixed=True,
        T_ic=fluid_T_ic,
        T_ic_fixed=True,
        k=1e5,
        enable_enthalpy_sensor=True,
    )
    pipe = fld.SimplePipe(ev, name="pipe", R=300.0, enable_sensors=True)
    if compX == "amb":
        comp_x = fld.Boundary_pT(ev, name="amb", p_ambient=p_ic)
    elif compX == "acc":
        comp_x = fld.Accumulator(
            ev,
            name="accum2",
            P_ic=p_ic,
            P_ic_fixed=True,
            T_ic=fluid_T_ic,
            T_ic_fixed=True,
            k=1e5,
        )
    else:
        comp_x = fld.OpenTank(
            ev,
            name="ot",
            P_amb=p_ic,
            T_ic=fluid_T_ic,
            T_ic_fixed=True,
            area=0.05,
            enabble_h_sensor=True,
        )
    sensPT = fld.PTSensor(ev, name="sensPT", enable_port_b=False)
    ad.connect(accum1, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", comp_x, "port")
    ad.connect(fp, "prop", accum1, "port")
    ad.connect(sensPT, "port_a", accum1, "port")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True, scale=True)
    acausal_system = ac()

    t_end = 10.0
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    pipeM_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("m_flow")]
    pipePa_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("pa")]
    pipePb_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("pb")]
    sensP_idx = acausal_system.outsym_to_portid[sensPT.get_sym_by_port_name("p")]
    sensT_idx = acausal_system.outsym_to_portid[sensPT.get_sym_by_port_name("temp")]
    acch_idx = acausal_system.outsym_to_portid[accum1.get_sym_by_port_name("h_output")]
    acchIn_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("sens_h_inStream")
    ]
    fldT_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("T_fluid_out")
    ]
    acc_port_mflow_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("port_mflow")
    ]
    acc_mass_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("sens_mass")
    ]
    acc_U_idx = acausal_system.outsym_to_portid[accum1.get_sym_by_port_name("sens_U")]
    acc_u_idx = acausal_system.outsym_to_portid[accum1.get_sym_by_port_name("sens_u")]
    recorded_signals = {
        "pipeM": acausal_system.output_ports[pipeM_idx],
        "pipePa": acausal_system.output_ports[pipePa_idx],
        "pipePb": acausal_system.output_ports[pipePb_idx],
        "sensP": acausal_system.output_ports[sensP_idx],
        "sensT": acausal_system.output_ports[sensT_idx],
        "acch": acausal_system.output_ports[acch_idx],
        "acchIn": acausal_system.output_ports[acchIn_idx],
        "acc_fldT": acausal_system.output_ports[fldT_idx],
        "acc_port_mflow": acausal_system.output_ports[acc_port_mflow_idx],
        "acc_mass": acausal_system.output_ports[acc_mass_idx],
        "acc_U": acausal_system.output_ports[acc_U_idx],
        "acc_u": acausal_system.output_ports[acc_u_idx],
    }
    if compX == "open_tank":
        height_idx = acausal_system.outsym_to_portid[
            comp_x.get_sym_by_port_name("height_output")
        ]
        recorded_signals["height"] = acausal_system.output_ports[height_idx]

    results = collimator.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals=recorded_signals,
    )
    t = results.time
    pipeM = results.outputs["pipeM"]
    pipePa = results.outputs["pipePa"]
    pipePb = results.outputs["pipePb"]
    sensP = results.outputs["sensP"]
    sensT = results.outputs["sensT"]
    acch = results.outputs["acch"]
    # acchIn = results.outputs["acchIn"]
    acc_fldT = results.outputs["acc_fldT"]
    acc_port_mflow = results.outputs["acc_port_mflow"]
    acc_mass = results.outputs["acc_mass"]
    acc_U = results.outputs["acc_U"]
    acc_u = results.outputs["acc_u"]
    if show_plot:
        fig, ((ax, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
            4, 2, figsize=(10, 10)
        )
        ax.plot(t, pipeM, label="pipeM")
        ax.plot(t, acc_port_mflow, label="acc_port_mflow")
        ax.legend()

        ax2.plot(t, pipePa, label="pipePa", linewidth=3)
        ax2.plot(t, pipePb, label="pipePb", linewidth=3)
        ax2.plot(t, sensP, label="sensP")
        ax2.legend()

        if compX == "open_tank":
            ax3.plot(t, results.outputs["height"], label="height")
        else:
            ax3.plot(t, sensT, label="sensPT h_inStream", linewidth=3)
            ax3.plot(t, acch, label="accum1 h_outflow")
            # ax3.plot(t, acchIn, label="accum1 h_inStream")
        ax3.legend()

        # ax4.plot(t, acc_mass, label="accum1 mass")
        ax4.plot(t, acc_U / acc_mass, label="accum1 U/mass")
        # ax4.plot(t, acc_u, label="acc_u")
        ax4.legend()

        ax5.plot(t, acc_U, label="accum1 U")
        ax5.legend()

        ax6.plot(t, acc_fldT, label="accum1 fluid temp")
        ax6.legend()

        ax7.plot(t, acc_mass, label="accum1 mass")
        ax7.legend()

        ax8.plot(t, acc_u, label="acc_u")
        ax8.legend()
        plt.show()


# Cross domain tests fluid(water)<->thermal
@pytest.mark.parametrize("use_heat_source", [True, False])
def test_heat_acc_pipe_amb(use_heat_source, show_plot=False):
    """
    A simple model demonstrating heat exchange into/out of incompressible fluid.

    heat_capacitor<->accumulator<->pipe<->amb.

    Initially the tank pressure is greater than ambient, which results
    in mass flow from tank to ambient. when the pressure difference gets
    to zero, the system comes to rest except for the heat flowing into
    the accumulator, increasing the temperature of the liquid inside it.

    FIXME: why does h_outflow of the accumulaor INCREASE during the simulation.
    sensT increases, which is expected since heat energy should flow from the
    warmer fluid to the cooler chiller. This thermal energy flow should result
    in h_outflow from the accumulator decreasing with time.
    """
    # build acausal model
    p_ic = P_DEFAULT
    fluid_T_ic = T_DEFAULT + 100  # deg K
    chiller_T_ic = T_DEFAULT + 200  # deg K
    ev = EqnEnv()
    fp = fld.FluidProperties(ev, fluid=fm.WaterLiquidSimple(ev))
    ad = AcausalDiagram()
    if use_heat_source:
        heat = thermal.HeatflowSource(
            ev,
            name="hs",
            Q_flow=1,  # small power because CV has small mass.
            enable_port_b=False,
        )
    else:
        heat = thermal.HeatCapacitor(
            ev,
            name="hc",
            initial_temperature=chiller_T_ic,
            initial_temperature_fixed=True,
            C=1e9,
        )
    sensT = thermal.TemperatureSensor(ev, name="sensT", enable_port_b=False)
    accum1 = fld.Accumulator(
        ev,
        name="accum1",
        P_ic=p_ic + 100,
        P_ic_fixed=True,
        T_ic=fluid_T_ic,
        T_ic_fixed=True,
        k=1e5,
        area=0.1,
        ht_coeff=100.0,
        enable_enthalpy_sensor=True,
        enable_thermal_port=True,
    )
    pipe = fld.SimplePipe(
        ev,
        name="pipe",
        R=1000.0,
        enable_sensors=True,
    )
    amb = fld.Boundary_pT(ev, name="amb", p_ambient=p_ic, T_ambient=fluid_T_ic)
    if use_heat_source:
        ad.connect(heat, "port_a", accum1, "wall")
    else:
        ad.connect(heat, "port", accum1, "wall")
    ad.connect(sensT, "port_a", accum1, "wall")
    ad.connect(accum1, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", amb, "port")
    ad.connect(fp, "prop", accum1, "port")
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()
    # run sim
    t_end = 10.0
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    pipeM_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("m_flow")]
    pipePa_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("pa")]
    pipePb_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("pb")]
    acch_idx = acausal_system.outsym_to_portid[accum1.get_sym_by_port_name("h_output")]
    sensT_idx = acausal_system.outsym_to_portid[sensT.get_sym_by_port_name("T_rel")]
    fldT_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("T_fluid_out")
    ]
    acc_port_mflow_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("port_mflow")
    ]
    sens_Qwall_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("sens_Qwall")
    ]
    acc_mass_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("sens_mass")
    ]
    acc_U_idx = acausal_system.outsym_to_portid[accum1.get_sym_by_port_name("sens_U")]
    recorded_signals = {
        "pipeM": acausal_system.output_ports[pipeM_idx],
        "pipePa": acausal_system.output_ports[pipePa_idx],
        "pipePb": acausal_system.output_ports[pipePb_idx],
        "acch": acausal_system.output_ports[acch_idx],
        "sensT": acausal_system.output_ports[sensT_idx],
        "acc_fldT": acausal_system.output_ports[fldT_idx],
        "acc_port_mflow": acausal_system.output_ports[acc_port_mflow_idx],
        "sens_Qwall": acausal_system.output_ports[sens_Qwall_idx],
        "acc_mass": acausal_system.output_ports[acc_mass_idx],
        "acc_U": acausal_system.output_ports[acc_U_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals=recorded_signals,
    )
    # plot results
    t = results.time
    pipeM = results.outputs["pipeM"]
    pipePa = results.outputs["pipePa"]
    pipePb = results.outputs["pipePb"]
    acch = results.outputs["acch"]
    sensT = results.outputs["sensT"]
    acc_fldT = results.outputs["acc_fldT"]
    acc_port_mflow = results.outputs["acc_port_mflow"]
    sens_Qwall = results.outputs["sens_Qwall"]
    acc_mass = results.outputs["acc_mass"]
    acc_U = results.outputs["acc_U"]
    if show_plot:
        fig, ((ax, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
            4, 2, figsize=(8, 8)
        )
        ax.plot(t, pipeM, label="pipeM")
        ax.plot(t, acc_port_mflow, label="acc_port_mflow")
        ax.legend()
        ax.grid()

        ax2.plot(t, pipePa, label="pipePa")
        ax2.plot(t, pipePb, label="pipePb")
        ax2.legend()
        ax2.grid()

        ax3.plot(t, acch, label="Accum h_outflow")
        ax3.legend()
        ax3.grid()

        ax4.plot(t, sensT, label="sensT wall")
        ax4.legend()
        ax4.grid()

        ax5.plot(t, sens_Qwall, label="sens Qwall")
        ax5.legend()
        ax5.grid()

        ax6.plot(t, acc_fldT, label="sensT accum fluid")
        ax6.legend()
        ax6.grid()

        ax7.plot(t, acc_mass, label="acc_mass")
        ax7.legend()
        ax7.grid()

        ax8.plot(t, acc_U, label="acc_U")
        ax8.legend()
        ax8.grid()
        plt.show()


@pytest.mark.skip(reason="might be flakey")
def test_acc_pipethermal_acc(show_plot=False):
    """
    The simplefluid model that has heat transfer between thermalmass and fluid.

    accum1->|
            |-pipethermal<->heat_capacitor
    accum2<-|

    where:
        accum1 is normal tempertaure high pressure accumulator acting as flow source
        accum2 is normal tempertaure low pressure accumulator acting as flow sink
    The expected outcome is that heat_capacitor adds thermal energy to the fluid in
    pipe, causing the tempertaure in accum2 to increase.
    """
    p_ic = 101325.0
    ev = EqnEnv()
    fp = fld.FluidProperties(ev, fluid=fm.WaterLiquidSimple(ev))
    ad = AcausalDiagram()

    p_ic = P_DEFAULT
    fluid_T_ic = T_DEFAULT  # deg K
    chiller_T_ic = T_DEFAULT + 200  # deg K

    accum1 = fld.Accumulator(
        ev,
        name="accum1",
        P_ic=p_ic + 100,
        P_ic_fixed=True,
        T_ic=fluid_T_ic,
        T_ic_fixed=True,
        k=1e5,
        area=0.1,
        enable_enthalpy_sensor=True,
    )
    pipe = fld.ThermalPipe(
        ev,
        name="pipe",
        R=10000.0,
        enable_sensors=True,
        ht_coeff=100,
        A=0.01,
        L=1.0,
    )
    accum2 = fld.Accumulator(
        ev,
        name="accum2",
        P_ic=p_ic,
        P_ic_fixed=True,
        T_ic=fluid_T_ic,
        T_ic_fixed=True,
        k=1e5,
        area=0.1,
        enable_enthalpy_sensor=True,
    )
    chiller = thermal.HeatCapacitor(
        ev,
        name="chiller",
        initial_temperature=chiller_T_ic,
        initial_temperature_fixed=True,
        C=1e6,
    )
    sensT = thermal.TemperatureSensor(ev, name="sensT", enable_port_b=False)
    ad.connect(accum1, "port", pipe, "port_a")
    ad.connect(pipe, "port_b", accum2, "port")
    ad.connect(fp, "prop", accum1, "port")
    ad.connect(pipe, "wall", chiller, "port")
    ad.connect(sensT, "port_a", chiller, "port")

    # compile to acausal system
    ac = AcausalCompiler(ev, ad, verbose=True)
    acausal_system = ac()

    t_end = 10.0
    builder = collimator.DiagramBuilder()
    acausal_system = builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context(check_types=True)
    pipeM_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("m_flow")]
    pipePa_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("pa")]
    pipePb_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("pb")]

    pipeU_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("sensU")]
    pipeQ_idx = acausal_system.outsym_to_portid[pipe.get_sym_by_port_name("sensQ")]
    pipeHin1_idx = acausal_system.outsym_to_portid[
        pipe.get_sym_by_port_name("sensHin1")
    ]
    pipeTin1_idx = acausal_system.outsym_to_portid[
        pipe.get_sym_by_port_name("sensTin1")
    ]

    sensT_idx = acausal_system.outsym_to_portid[sensT.get_sym_by_port_name("T_rel")]
    acc1_hout_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("h_output")
    ]
    acc1_T_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("T_fluid_out")
    ]
    acc1_mass_idx = acausal_system.outsym_to_portid[
        accum1.get_sym_by_port_name("sens_mass")
    ]
    acc1_U_idx = acausal_system.outsym_to_portid[accum1.get_sym_by_port_name("sens_U")]
    acc2_hout_idx = acausal_system.outsym_to_portid[
        accum2.get_sym_by_port_name("h_output")
    ]
    acc2_T_idx = acausal_system.outsym_to_portid[
        accum2.get_sym_by_port_name("T_fluid_out")
    ]
    acc2_mass_idx = acausal_system.outsym_to_portid[
        accum2.get_sym_by_port_name("sens_mass")
    ]
    acc2_U_idx = acausal_system.outsym_to_portid[accum2.get_sym_by_port_name("sens_U")]

    recorded_signals = {
        "pipeM": acausal_system.output_ports[pipeM_idx],
        "pipePa": acausal_system.output_ports[pipePa_idx],
        "pipePb": acausal_system.output_ports[pipePb_idx],
        "pipeU": acausal_system.output_ports[pipeU_idx],
        "pipeQ": acausal_system.output_ports[pipeQ_idx],
        "pipeHin1": acausal_system.output_ports[pipeHin1_idx],
        "pipeTin1": acausal_system.output_ports[pipeTin1_idx],
        "sensT": acausal_system.output_ports[sensT_idx],
        "acc1_hout": acausal_system.output_ports[acc1_hout_idx],
        "acc1_T": acausal_system.output_ports[acc1_T_idx],
        "acc1_mass": acausal_system.output_ports[acc1_mass_idx],
        "acc1_U": acausal_system.output_ports[acc1_U_idx],
        "acc2_hout": acausal_system.output_ports[acc2_hout_idx],
        "acc2_T": acausal_system.output_ports[acc2_T_idx],
        "acc2_mass": acausal_system.output_ports[acc2_mass_idx],
        "acc2_U": acausal_system.output_ports[acc2_U_idx],
    }
    results = collimator.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals=recorded_signals,
    )
    t = results.time
    pipeM = results.outputs["pipeM"]
    pipePa = results.outputs["pipePa"]
    pipePb = results.outputs["pipePb"]
    pipeU = results.outputs["pipeU"]
    pipeQ = results.outputs["pipeQ"]
    pipeHin1 = results.outputs["pipeHin1"]
    pipeTin1 = results.outputs["pipeTin1"]

    sensT = results.outputs["sensT"]

    acc1_hout = results.outputs["acc1_hout"]
    acc1_T = results.outputs["acc1_T"]
    acc1_mass = results.outputs["acc1_mass"]
    acc1_U = results.outputs["acc1_U"]

    acc2_hout = results.outputs["acc2_hout"]
    acc2_T = results.outputs["acc2_T"]
    acc2_mass = results.outputs["acc2_mass"]
    acc2_U = results.outputs["acc2_U"]

    if show_plot:
        fig, ((ax, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(
            4, 2, figsize=(10, 10)
        )
        ax.plot(t, pipeM, label="pipeM")
        ax.grid()
        ax.legend()

        ax2.plot(t, pipePa, label="pipePa", linewidth=3)
        ax2.plot(t, pipePb, label="pipePb", linewidth=3)
        ax2.grid()
        ax2.legend()

        # ax3.plot(t, sensT, label="sensT")
        ax3.plot(t, acc1_T, label="acc1_T", linewidth=3)
        ax3.plot(t, acc2_T, label="acc2_T")
        ax3.plot(t, pipeTin1, label="pipeTin1")
        ax3.grid()
        ax3.legend()

        ax4.plot(t, acc1_mass, label="acc1_mass")
        ax4.plot(t, acc2_mass, label="acc2_mass")
        ax4.grid()
        ax4.legend()

        ax5.plot(t, acc1_U, label="acc1_U")
        ax5.plot(t, acc2_U, label="acc2_U")
        # ax5.plot(t, pipeU, label="pipeU")
        ax5.grid()
        ax5.legend()

        ax6.plot(t, acc1_hout, label="acc1_hout", linewidth=3)
        ax6.plot(t, acc2_hout, label="acc2_hout")
        ax6.plot(t, pipeHin1, label="pipeHin1")
        ax6.grid()
        ax6.legend()

        ax7.plot(t, pipeQ, label="pipeQ")
        ax7.grid()
        ax7.legend()

        # ax8.plot(t, sensT, label="sensT")
        ax8.plot(t, pipeU, label="pipeU")
        ax8.grid()
        ax8.legend()
        plt.show()


if __name__ == "__main__":
    show_plot = True
    # test_fluid_inline_pump(show_plot=show_plot)
    # test_hyd_act_and_spring(show_plot=show_plot)
    # test_tank_pipe_amb(False, show_plot=show_plot)
    # test_tank_pipe_amb(True, show_plot=show_plot)
    # test_tank_pipe_amb_with_sensors(show_plot=show_plot)
    # test_T_junction_splitting(show_plot=show_plot)
    # test_T_junction_merging(show_plot=show_plot)
    # test_heat_tank_pipe_amb(False, show_plot=show_plot)
    # test_heat_tank_pipe_amb(True, show_plot=show_plot)
    # test_water_if97_region1()
    test_acc_pipe_compX("amb", show_plot=show_plot)
    # test_acc_pipe_compX("acc", show_plot=show_plot)
    # test_acc_pipe_compX("open_tank", show_plot=show_plot)
    # test_heat_acc_pipe_amb(False, show_plot=show_plot)
    # test_heat_acc_pipe_amb(True, show_plot=show_plot)
    # test_acc_pipethermal_acc(show_plot=show_plot)
