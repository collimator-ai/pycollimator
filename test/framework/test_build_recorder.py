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

from collimator import DiagramBuilder, Parameter, library, simulate
from collimator.experimental import AcausalCompiler, AcausalDiagram, EqnEnv
from collimator.experimental import translational as trans
from collimator.experimental import electrical
from collimator.framework import build_recorder
import collimator.testing as testing


def test_build_recorder_simple():
    build_recorder.start()
    builder = DiagramBuilder()
    c = Parameter(name="c", value=np.array([1.0, 2.0]))
    g = Parameter(name="g", value=2.0)
    p = Parameter(name="p", value=3.0)
    Constant_0 = builder.add(library.Constant(value=c))
    builder.add(library.Constant(value=np.array([1.0, 2.0])))
    Gain_0 = builder.add(library.Gain(gain=g * p, name="Gain_0"))
    builder.connect(Constant_0.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build("root", parameters={"c": c, "g": g, "p": p})
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={"Gain_0.out_0": Gain_0.output_ports[0]},
    )

    np.testing.assert_array_equal(results.outputs["Gain_0.out_0"][0], [6.0, 12.0])

    code = build_recorder.generate_code()
    build_recorder.stop()

    exec(code, globals())

    new_diagram = globals()["root"]
    Gain_0 = globals()["root_Gain_0"]
    new_context = new_diagram.create_context()
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={"Gain_0.out_0": Gain_0.output_ports[0]},
    )

    np.testing.assert_array_equal(new_results.outputs["Gain_0.out_0"][0], [6.0, 12.0])


def test_build_recorder_groups():
    build_recorder.start()
    builder = DiagramBuilder()
    c = Parameter(name="c", value=np.array([1.0, 2.0]))
    g = Parameter(name="g", value=3.0)

    Gain_0 = builder.add(library.Gain(gain=1 + g, name="Gain_0"))

    def _make_group():
        group_builder = DiagramBuilder()
        Gain_0 = group_builder.add(library.Gain(gain=g, name="Gain_0"))
        group_builder.export_input(Gain_0.input_ports[0], "in_0")
        group_builder.export_output(Gain_0.output_ports[0], "out_0")
        group = group_builder.build("group")
        return group

    group_diagram = _make_group()
    builder.add(group_diagram)
    constant = builder.add(library.Constant(value=c))
    builder.connect(constant.output_ports[0], group_diagram.input_ports[0])
    builder.connect(constant.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build("root", parameters={"c": c, "g": g})
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "group.out_0": group_diagram.output_ports[0],
            "Gain_0.out_0": Gain_0.output_ports[0],
        },
    )

    np.testing.assert_array_equal(results.outputs["group.out_0"][0], [3.0, 6.0])
    np.testing.assert_array_equal(results.outputs["Gain_0.out_0"][0], [4.0, 8.0])

    code = build_recorder.generate_code()
    build_recorder.stop()

    exec(code, globals())

    new_diagram = globals()["root"]
    group = globals()["root_group"]
    gain = globals()["root_Gain_0"]
    new_context = new_diagram.create_context()
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={
            "group.out_0": group.output_ports[0],
            "gain.out_0": gain.output_ports[0],
        },
    )

    np.testing.assert_array_equal(new_results.outputs["group.out_0"][0], [3.0, 6.0])
    np.testing.assert_array_equal(new_results.outputs["gain.out_0"][0], [4.0, 8.0])


def test_build_recorder_submodels():
    # FIXME: reference submodels are generated as groups.
    build_recorder.start()
    builder = DiagramBuilder()
    c = Parameter(name="c", value=np.array([1.0, 2.0]))
    g = Parameter(name="g", value=3.0)

    Gain_0 = builder.add(library.Gain(gain=g, name="Gain_0"))

    def _make_submodel(instance_name, parameters):
        sub_builder = DiagramBuilder()
        Gain_0 = sub_builder.add(library.Gain(gain=parameters["gain"], name="Gain_0"))
        sub_builder.export_input(Gain_0.input_ports[0], "in_0")
        sub_builder.export_output(Gain_0.output_ports[0], "out_0")
        sub = sub_builder.build(instance_name)
        return sub

    ref_id = library.ReferenceSubdiagram.register(
        _make_submodel, parameter_definitions=[Parameter(name="gain", value=0)]
    )

    sub1 = library.ReferenceSubdiagram.create_diagram(
        ref_id, "sub1", instance_parameters={"gain": g + 1}
    )
    sub2 = library.ReferenceSubdiagram.create_diagram(
        ref_id, "sub2", instance_parameters={"gain": g + 2}
    )

    builder.add(sub1)
    builder.add(sub2)
    constant = builder.add(library.Constant(value=c))

    builder.connect(constant.output_ports[0], sub1.input_ports[0])
    builder.connect(constant.output_ports[0], sub2.input_ports[0])
    builder.connect(constant.output_ports[0], Gain_0.input_ports[0])

    diagram = builder.build("root", parameters={"c": c, "g": g})
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "sub1.out_0": sub1.output_ports[0],
            "sub2.out_0": sub2.output_ports[0],
            "Gain_0.out_0": Gain_0.output_ports[0],
        },
    )

    np.testing.assert_array_equal(results.outputs["sub1.out_0"][0], [4.0, 8.0])
    np.testing.assert_array_equal(results.outputs["sub2.out_0"][0], [5.0, 10.0])
    np.testing.assert_array_equal(results.outputs["Gain_0.out_0"][0], [3.0, 6.0])

    code = build_recorder.generate_code()
    build_recorder.stop()

    exec(code, globals())

    new_diagram = globals()["root"]
    sub1 = globals()["root_sub1"]
    sub2 = globals()["root_sub2"]
    gain = globals()["root_Gain_0"]
    new_context = new_diagram.create_context()
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={
            "sub1.out_0": sub1.output_ports[0],
            "sub2.out_0": sub2.output_ports[0],
            "gain.out_0": gain.output_ports[0],
        },
    )

    np.testing.assert_array_equal(new_results.outputs["sub1.out_0"][0], [4.0, 8.0])
    np.testing.assert_array_equal(new_results.outputs["sub2.out_0"][0], [5.0, 10.0])
    np.testing.assert_array_equal(new_results.outputs["gain.out_0"][0], [3.0, 6.0])


def test_build_recorder_python_script_block():
    build_recorder.start()
    builder = DiagramBuilder()

    PythonScriptBlock = builder.add(
        library.CustomPythonBlock(
            dt=0.1,
            user_statements="import numpy as np\nout_0 = in_0 * 2",
            init_script="out_0 = 0.0",
            inputs=["in_0"],
            outputs=["out_0"],
            name="PythonScriptBlock",
        )
    )
    Constant = builder.add(library.Constant(value=1))
    builder.connect(Constant.output_ports[0], PythonScriptBlock.input_ports[0])
    diagram = builder.build("root")
    context = diagram.create_context()

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={"PythonScriptBlock.out_0": PythonScriptBlock.output_ports[0]},
    )

    assert results.outputs["PythonScriptBlock.out_0"][-1] == 2.0

    code = build_recorder.generate_code()
    build_recorder.stop()

    print(code)
    exec(code, globals())

    new_diagram = globals()["root"]
    PythonScriptBlock = globals()["root_PythonScriptBlock"]
    new_context = new_diagram.create_context()
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={"PythonScriptBlock.out_0": PythonScriptBlock.output_ports[0]},
    )

    assert new_results.outputs["PythonScriptBlock.out_0"][-1] == 2.0


@testing.requires_jax()
def test_acausal_model_build():
    build_recorder.start()
    ev = EqnEnv()
    ad = AcausalDiagram()

    spring_k = Parameter(name="k", value=1.0)
    m1 = trans.Mass(
        ev,
        name="m1",
        M=1.0,
        initial_position=1.0,
        initial_position_fixed=True,
        initial_velocity=0.0,
        initial_velocity_fixed=True,
    )
    sp1 = trans.Spring(ev, name="sp1", K=spring_k)
    r1 = trans.FixedPosition(ev, name="r1", initial_position=0.0)
    spdsnsr1 = trans.MotionSensor(
        ev,
        name="spdsnsr",
        enable_flange_b=True,
        enable_position_port=True,
    )
    ad.connect(m1, "flange", sp1, "flange_a")
    ad.connect(sp1, "flange_b", r1, "flange")
    ad.connect(m1, "flange", spdsnsr1, "flange_a")
    ad.connect(r1, "flange", spdsnsr1, "flange_b")

    ac = AcausalCompiler(ev, ad)
    acausal_system = ac(leaf_backend="jax")

    builder = DiagramBuilder()
    builder.add(acausal_system)
    diagram = builder.build()
    context = diagram.create_context()

    x_idx = acausal_system.outsym_to_portid[spdsnsr1.get_sym_by_port_name("x_rel")]

    results = simulate(
        diagram,
        context,
        (0.0, 1.0),
        recorded_signals={
            "x": acausal_system.output_ports[x_idx],
        },
    )
    x_sol = np.cos(results.time)
    assert np.allclose(results.outputs["x"], x_sol, rtol=0.0, atol=1e-2)

    code = build_recorder.generate_code()
    build_recorder.stop()

    print(code)
    exec(code, globals())

    new_diagram = globals()["root"]
    new_acausal_system = globals()["root_acausal_system"]
    new_spdsnsr = globals()["spdsnsr"]
    new_context = new_diagram.create_context()

    new_x_idx = new_acausal_system.outsym_to_portid[
        new_spdsnsr.get_sym_by_port_name("x_rel")
    ]
    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={"x": new_acausal_system.output_ports[new_x_idx]},
    )

    new_x_sol = np.cos(new_results.time)
    assert np.allclose(new_results.outputs["x"], new_x_sol, rtol=0.0, atol=1e-2)


@testing.requires_jax()
def test_acausal_model_build_with_groups():
    build_recorder.start()
    group_builder = DiagramBuilder()
    root_builder = DiagramBuilder()

    ee = EqnEnv()
    VoltageSource = electrical.VoltageSource(
        ee, name="VoltageSource", v=5.0, enable_voltage_port=True
    )
    Ground = electrical.Ground(ee, name="Ground")
    Resistor = electrical.Resistor(ee, name="Resistor", R=10.0)
    Capacitor = electrical.Capacitor(ee, name="Capacitor", C=20.0, initial_voltage=0.0)
    VoltageSensor = electrical.VoltageSensor(ee, name="VoltageSensor")
    ad = AcausalDiagram()

    ad.connect(Resistor, "p", VoltageSource, "p")
    ad.connect(Resistor, "n", Capacitor, "p")
    ad.connect(Capacitor, "n", VoltageSource, "n")
    ad.connect(VoltageSource, "n", Ground, "p")
    ad.connect(VoltageSensor, "p", Resistor, "p")
    ad.connect(Resistor, "n", VoltageSensor, "n")
    ac = AcausalCompiler(ee, ad)

    acausal_system = ac.generate_acausal_system(name="root_Group_0_acausal_system")
    group_builder.add(acausal_system)
    Group_0_Outport_0 = library.IOPort()
    group_builder.add(Group_0_Outport_0)
    Group_0_Inport_0 = library.IOPort()
    group_builder.add(Group_0_Inport_0)
    group_builder.export_input(Group_0_Inport_0.input_ports[0], "Inport_0")
    group_builder.export_output(Group_0_Outport_0.output_ports[0], "Outport_0")
    group_builder.connect(
        acausal_system.output_ports[0], Group_0_Outport_0.input_ports[0]
    )
    group_builder.connect(
        Group_0_Inport_0.output_ports[0], acausal_system.input_ports[0]
    )

    root_Group_0 = group_builder.build("Group_0")
    root_builder.add(root_Group_0)
    root_Constant_0 = library.Constant(10.0)
    root_builder.add(root_Constant_0)
    root_builder.connect(root_Constant_0.output_ports[0], root_Group_0.input_ports[0])
    root = root_builder.build("root")

    context = root.create_context()

    results = simulate(
        root,
        context,
        (0.0, 1.0),
        recorded_signals={
            "x": root_Group_0.output_ports[0],
        },
    )
    x_sol1 = results.outputs["x"]

    code = build_recorder.generate_code()
    build_recorder.stop()

    print(code)
    exec(code, globals())

    new_diagram = globals()["root"]
    new_root_Group_0 = globals()["root_Group_0"]
    new_context = new_diagram.create_context()

    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={"x": new_root_Group_0.output_ports[0]},
    )

    x_sol2 = new_results.outputs["x"]

    assert np.allclose(x_sol1, x_sol2, rtol=0.0, atol=1e-2)


@testing.requires_jax()
def test_acausal_model_build_with_submodel():
    build_recorder.start()
    root_builder = DiagramBuilder()

    def _make_submodel(instance_name, parameters):
        group_builder = DiagramBuilder()
        ee = EqnEnv()
        VoltageSource = electrical.VoltageSource(
            ee, name="VoltageSource", v=5.0, enable_voltage_port=True
        )
        Ground = electrical.Ground(ee, name="Ground")
        Resistor = electrical.Resistor(ee, name="Resistor", R=10.0)
        Capacitor = electrical.Capacitor(
            ee, name="Capacitor", C=20.0, initial_voltage=0.0
        )
        VoltageSensor = electrical.VoltageSensor(ee, name="VoltageSensor")
        ad = AcausalDiagram()

        ad.connect(Resistor, "p", VoltageSource, "p")
        ad.connect(Resistor, "n", Capacitor, "p")
        ad.connect(Capacitor, "n", VoltageSource, "n")
        ad.connect(VoltageSource, "n", Ground, "p")
        ad.connect(VoltageSensor, "p", Resistor, "p")
        ad.connect(Resistor, "n", VoltageSensor, "n")
        ac = AcausalCompiler(ee, ad)

        acausal_system = ac.generate_acausal_system(name="acausal_system")
        group_builder.add(acausal_system)
        Group_0_Outport_0 = library.IOPort()
        group_builder.add(Group_0_Outport_0)
        Group_0_Inport_0 = library.IOPort()
        group_builder.add(Group_0_Inport_0)
        group_builder.export_input(Group_0_Inport_0.input_ports[0], "Inport_0")
        group_builder.export_output(Group_0_Outport_0.output_ports[0], "Outport_0")
        group_builder.connect(
            acausal_system.output_ports[0], Group_0_Outport_0.input_ports[0]
        )
        group_builder.connect(
            Group_0_Inport_0.output_ports[0], acausal_system.input_ports[0]
        )

        return group_builder.build(instance_name)

    ref_id = library.ReferenceSubdiagram.register(_make_submodel)
    sub1 = library.ReferenceSubdiagram.create_diagram(ref_id, "sub1")

    root_builder.add(sub1)
    root_Constant_0 = library.Constant(10.0)
    root_builder.add(root_Constant_0)
    root_builder.connect(root_Constant_0.output_ports[0], sub1.input_ports[0])
    root = root_builder.build("root")

    context = root.create_context()

    results = simulate(
        root,
        context,
        (0.0, 1.0),
        recorded_signals={
            "x": sub1.output_ports[0],
        },
    )
    x_sol1 = results.outputs["x"]

    code = build_recorder.generate_code()
    build_recorder.stop()

    print(code)
    exec(code, globals())

    new_diagram = globals()["root"]
    new_sub1 = globals()["root_sub1"]
    new_context = new_diagram.create_context()

    new_results = simulate(
        new_diagram,
        new_context,
        (0.0, 1.0),
        recorded_signals={"x": new_sub1.output_ports[0]},
    )

    x_sol2 = new_results.outputs["x"]

    assert np.allclose(x_sol1, x_sol2, rtol=0.0, atol=1e-2)
