from __future__ import annotations
import dataclasses
import glob
import io
import json
import os
from typing import Any, TYPE_CHECKING
import warnings

import jax.numpy as jnp
import numpy as np
import control
import math

from ..simulation import simulate, ODESolverOptions, SimulatorOptions, ResultsOptions
from ..framework import (
    CacheSource,
    Diagram,
    DiagramBuilder,
    SystemBase,
)
from ..library import ReferenceSubdiagram
from ..binary_results import write_binary_results_f, _map_types
from . import types as json_types
from . import block_interface
from lynx import logging

if TYPE_CHECKING:
    from ..framework import ContextBase

__all__ = [
    "loads_model",
    "load_model",
    "register_reference_submodel",
    "SimulationContext",
]

"""
Layer between the Collimator app JSON output and the sim engine.
Presently, this works for model.json+list[submodel-uuid-ver.json] content
from both simworkerpy(wildcat) and simworker(go)(cmlc).

Processing steps:
1] The classes in types.py are used to "read" the json, packaging convinently
Not ideal, but there is some transformation that happen here.
Specifically, model.json.parameters dict is transformed to match List[ParameterDefinition]
so that the top level diagram and all submodel diagrams can be treated identically
w.r.t. parameter namespace handling at later loading stages.

2] visit all the model subdiagrams by depth first search through the json.
create node name_path ids, and a dict diagrams[subdiagram_node_id]=LoadCtxtDiagram (e.g. most the raw json content).

3] visit all the elements of diagrams dict, and build a wildcat diagram object for each, composing them as we go.

"""


# Used for exception chaining to wrap a Python exception with the offending
# block id and parameter name
class BlockParamError(Exception):
    def __init__(self, block_id, param_name):
        self.block_id = block_id
        self.param_name = param_name


@dataclasses.dataclass
class SimulationContext:
    diagram: Diagram
    ode_solver_options: ODESolverOptions
    results_options: ResultsOptions
    recorded_signals: dict[str, CacheSource]
    simulator_options: SimulatorOptions = None


def loads_model(
    model_json: str,
    namespace: dict[str, Any] = None,
) -> SimulationContext:
    """Load a model from a JSON string.

    Reference submodels must be registered before calling this function.
    """
    if namespace is None:
        namespace = {}
    model_fp = io.StringIO(model_json)
    model = json_types.Model.from_json(model_fp)

    root_namespace = namespace
    model_parameters = eval_parameters(
        "root",
        default_parameters=model.parameter_definitions,
        instance_parameters=model.parameters,
        call_site_namespace=namespace,
    )
    root_namespace.update(model_parameters)

    recorded_signals = {}
    diagram = make_subdiagram(
        "root",
        model.diagram,
        model.subdiagrams,
        namespace_params=root_namespace,
        global_discrete_interval=model.configuration.sample_time,
        record_mode=model.configuration.record_mode,
        recorded_signals=recorded_signals,
    )

    ode_options, results_options = None, None
    if model.configuration:
        ode_options, results_options, simulator_options = simulation_settings(
            model.configuration, recorded_signals=recorded_signals
        )

    return SimulationContext(
        diagram=diagram,
        ode_solver_options=ode_options,
        results_options=results_options,
        recorded_signals=recorded_signals,
        simulator_options=simulator_options,
    )


def load_model(
    modeldir: str,
    model: str = "model.json",
    datadir: str = None,
    logsdir: str = None,
    block_overrides=None,
) -> AppInterface:
    # @am. eventually we'll wnat to NOT allow None for some or all of the dirs.

    # register reference submodels
    file_pattern = os.path.join(modeldir, "submodel-*-latest.json")
    submodel_files = glob.glob(file_pattern)
    for submodel_file in submodel_files:
        ref_id = os.path.basename(submodel_file).split("-")[1:-1]
        ref_id = "-".join(ref_id)
        with open(submodel_file, "r") as f:
            submodel = json_types.Model.from_json(f)
            register_reference_submodel(ref_id, submodel)

    with open(os.path.join(modeldir, model), "r") as f:
        model = json_types.Model.from_json(f)

    return AppInterface(
        model, datadir=datadir, logsdir=logsdir, block_overrides=block_overrides
    )


def eval_parameter(value: str, _globals: dict, _locals: dict):
    if not value:
        return None
    p = eval(str(value), _globals, _locals)
    # Rules for user-input parameters:
    # - if explicitly given as an array with dtype, use that dtype
    # - if boolean, use as given
    # - otherwise, convert to a numpy array
    if not hasattr(p, "dtype") and not isinstance(p, bool):
        p = np.array(p)
        # if the array has integer dtype convert to float. Note that
        # this case will still not be reached if the parameter is explicitly
        # declared as an array with integer datatype.  However, this will
        # promote inputs like `"0"` or `"[1, 0]"` to floats.
        if issubclass(p.dtype.type, np.integer):
            p = p.astype(float)
    return p


def eval_parameters(
    block_id: str,
    default_parameters: list[json_types.ParameterDefinition] = None,
    instance_parameters: dict[str, json_types.Parameter] = None,
    call_site_namespace: dict[str, Any] = None,
):
    # parameter handling.
    # at this point we have the following:
    # 1] diagrams[block_id].default_parameters
    #   these are from the submodel-uuid-ver.json[parameter_definitions].
    #   these are the definitions and defaults for all parameters in the name space of the submodel
    # 2] instance_parameters
    #   these are the parameter values from the instance of the submodel. this must be a subset of 1]
    #   this dict is typically limited to those params which have been modified from their default value.
    #   the values of these should have been evaluated in the context of the parent of the instance.
    # 3] call_site_namespace
    #   these are the parameters in the name space of the parent.
    #   these are only used here because before getting here, only the instance parameters
    #   have been evaluated in the parent name space. if a parameters has not been modified,
    #   then we use the default value from 1] but we need to evaluate it in the parent name space.
    #
    # Note: reference submodels have a 'protected' parameter names space. parents can only pass
    # parameter values through the submodels defined parameters. i.e. there is no shared name space
    # between the reference submodel instance and its parent.

    if default_parameters is None:
        default_parameters = []
    if call_site_namespace is None:
        call_site_namespace = {}
    if instance_parameters is None:
        instance_parameters = {}

    def _eval(param_name, value: str, _locals):
        try:
            return eval_parameter(value, globals(), _locals)
        # We will probably want to be much more discriminating with respect to what gets
        # caught here. This uses exception chaining to record the offending block id
        # and parameter name.
        except Exception as exc:
            logging.error("Error evaluating parameter %s. Error: %s", param_name, exc)
            raise BlockParamError(block_id, param_name) from exc

    # multi round eval is required in case we have parameters that reference other parameters.
    def _multi_round_eval(params, eval_fn, _locals):
        eval_results = {}
        max_eval_depth = 10
        for i in range(max_eval_depth):
            need_eval = False
            for pname, p in params.items():
                try:
                    eval_results[pname] = eval_fn(pname, p, _locals)
                except BlockParamError as exc:
                    if i == max_eval_depth - 1:
                        raise exc
                    need_eval = True
            _locals.update(eval_results)
            if not need_eval:
                break
        return eval_results

    # We don't technically need to pass np and jnp here, but it makes them explicitly
    # available in the local `eval` environment.
    _locals = {
        **call_site_namespace,
        "np": np,
        "jnp": jnp,
        "math": math,
        "false": False,
        "true": True,
    }

    def _eval_param_def(pname, p, _locals):
        if p.default_value != "":
            return _eval(pname, p.default_value, _locals)
        return None

    default_values = _multi_round_eval(
        {p.name: p for p in default_parameters},
        _eval_param_def,
        _locals,
    )
    instance_values = _multi_round_eval(
        instance_parameters,
        lambda pname, p, l: _eval(pname, p.value, l) if not p.is_string else p.value,
        _locals,
    )

    default_values.update(instance_values)
    return default_values


def eval_init_script(
    block_id: str, init_script_file_name: str, namespace_params: dict = {}
) -> dict:
    # We don't technically need to pass np and jnp here, but it makes them explicitly
    # available in the local `exec` environment. a user could also do these imports.
    # "__main__": {} is necessary for script of the form
    #   imports ...
    #   a = 1
    #   def f(b):
    #       return a+b
    #   out_0 = f(2)
    # withotu getting "a not defined" error.

    _locals = {
        **namespace_params,
        "np": np,
        "jnp": jnp,
        "control": control,
        "__main__": {},
    }

    with open(init_script_file_name, "r") as file:
        _init_script_code = file.read()

    try:
        exec(_init_script_code, _locals, _locals)
    except Exception as e:
        raise RuntimeError(
            f"failed to execute init_script {init_script_file_name} with {e}"
        )

    # _locals will be pretty messy after exec() since it wil contain all the stuff
    # python puts in globals for any execution. doesn't matter, this globals env
    # is only retained for parameter evaluation.
    return _locals


def GroupBlock(
    block_spec: json_types.Node,
    global_discrete_interval: float,
    subdiagrams: dict[str, json_types.Diagram],
    record_mode: str = "selected",
    namespace_params=None,
    block_overrides=None,
    block_path=None,
    recorded_signals=None,
):
    if namespace_params is None:
        namespace_params = {}
    if block_path is None:
        block_path = []

    diagram = make_subdiagram(
        block_spec.name,
        subdiagrams.get_diagram(block_spec.uuid),
        subdiagrams,
        global_discrete_interval=global_discrete_interval,
        namespace_params=namespace_params,
        block_overrides=block_overrides,
        block_path=block_path + [block_spec.name],
        record_mode=record_mode,
        recorded_signals=recorded_signals,
    )
    return diagram


def register_reference_submodel(ref_id: str, model: json_types.Model):
    def _make_subdiagram_instance(
        global_discrete_interval,
        block_name,
        parameters,
        block_path,
        record_mode,
        recorded_signals,
    ):
        namespace_params = {k: p.evaluated_value for k, p in parameters.items()}
        return make_subdiagram(
            block_name,
            model.diagram,
            model.subdiagrams,
            namespace_params=namespace_params,
            block_path=block_path,
            global_discrete_interval=global_discrete_interval,
            record_mode=record_mode,
            recorded_signals=recorded_signals,
        )

    parameters = None
    if model.parameter_definitions:
        parameters = model.parameter_definitions
    elif model.parameters:
        # NOTE: simworker-go instantiates the submodel with parameters (dict of parameters)
        # see test_0080a.py for example.
        # So here we convert to parameter definitions (list of ParameterDefinition).
        logging.debug("Converting instantiated parameters to parameter definitions.")
        parameters = [
            json_types.ParameterDefinition(name=k, default_value=v.value)
            for k, v in model.parameters.items()
        ]

    ReferenceSubdiagram.register(
        ref_id,
        _make_subdiagram_instance,
        parameter_definitions=parameters,
    )


# FIXME: simplify this function. it's too long.
# pylint: disable=too-complex
# flake8: noqa: C901
def make_subdiagram(
    name: str,
    diagram: json_types.Diagram,
    subdiagrams: dict[str, json_types.Diagram],
    namespace_params=None,
    block_overrides: dict[str, SystemBase] = None,
    block_path=None,
    record_mode: str = "selected",
    recorded_signals: dict[str, CacheSource] = None,
    global_discrete_interval: float = 0.1,
) -> Diagram:
    if namespace_params is None:
        namespace_params = {}
    if block_path is None:
        block_path = []
    # The "node_spec" passed here is the "node" specification that doesn't
    # contain the actual blocks. the info about the block tat constains the
    # subdiagram, e.g. submodel instance.

    # TODO: correctly handle RefSubmodelConfiguration features
    # (atomic, discrete_step)

    builder = DiagramBuilder()

    # I/O ports are considered "blocks" in the UI, so they need to be tracked
    # specificially
    exported_inputs = []
    exported_outputs = []

    # needed for dereferencing node ids in link specs. this map is local to a
    # canvas.
    uuid_to_block_id = {}

    blocks = {}

    # print(f"\nMaking subdiagram: {name}")

    for block_spec in diagram.nodes:
        block_id = block_spec.name
        uuid_to_block_id[block_spec.uuid] = block_id
        if block_spec.type == "core.Inport":
            exported_inputs.append(block_id)
        elif block_spec.type == "core.Outport":
            exported_outputs.append(block_id)

        # eval io port parameters
        io_ports_params = {}
        for io_port in block_spec.outputs + block_spec.inputs:
            io_ports_params[io_port.name] = {}
            if io_port.parameters:
                io_ports_params[io_port.name] = eval_parameters(
                    block_spec.name,
                    instance_parameters=io_port.parameters,
                    call_site_namespace=namespace_params,
                )

        if block_overrides and block_id in block_overrides:
            block = block_overrides[block_id]
        elif block_spec.type == "core.ReferenceSubmodel":
            logging.debug(
                "Creating reference submodel %s (ref id: %s) "
                "with instance_parameters: %s, call_site_namespace: %s",
                block_id,
                block_spec.submodel_reference_uuid,
                block_spec.parameters,
                namespace_params,
            )

            instance_parameters = {k: p.value for k, p in block_spec.parameters.items()}
            block = ReferenceSubdiagram.create_diagram(
                block_spec.submodel_reference_uuid,
                global_discrete_interval=global_discrete_interval,
                record_mode=record_mode,
                block_name=block_id,
                call_site_namespace=namespace_params,
                instance_parameters=instance_parameters,
                block_path=block_path + [block_id],
                recorded_signals=recorded_signals,
            )
        elif block_spec.type in ("core.Group", "core.Submodel"):
            block = GroupBlock(
                block_spec,
                global_discrete_interval,
                subdiagrams,
                namespace_params=namespace_params,
                block_path=block_path,
                record_mode=record_mode,
                recorded_signals=recorded_signals,
            )
        else:
            common_kwargs = {
                "name": block_id,
                "system_id": ".".join(block_path + [block_id]),
                "io_ports_params": io_ports_params,
            }
            parameters = eval_parameters(
                block_id,
                instance_parameters=block_spec.parameters,
                call_site_namespace=namespace_params,
            )
            block = block_interface.get_block_fcn(block_spec.type)(
                block_spec=block_spec,
                discrete_interval=global_discrete_interval,
                **common_kwargs,
                **parameters,
            )

            if block_spec.inputs:
                input_port_names = [port.name for port in block_spec.inputs]
                for port_name, port in zip(input_port_names, block.input_ports):
                    port.name = port_name

            if block_spec.outputs:
                output_port_names = [port.name for port in block_spec.outputs]
                for port_name, port in zip(output_port_names, block.output_ports):
                    port.name = port_name

        builder.add(block)
        blocks[block_id] = block
        # NOTE: Here we assume that the port order is the same in the frontend and wildcat
        # Log anything with record=True
        if block_spec.outputs and recorded_signals is not None:
            for i, port in enumerate(block_spec.outputs):
                if port.record or record_mode == "all":
                    port_path = block_path + [block_id, port.name]
                    port_path = ".".join(port_path)
                    logging.debug("Recording %s", port_path)
                    recorded_signals[port_path] = block.output_ports[i]

    # Export the input port of any Inport
    for input_port_id_key in exported_inputs:
        builder.export_input(blocks[input_port_id_key].input_ports[0])

    # Export the output port of any Outport
    for output_port_id_key in exported_outputs:
        builder.export_output(blocks[output_port_id_key].output_ports[0])

    for link in diagram.links:
        if (
            (link.src is None)
            or (link.dst is None)
            or (link.src.node not in uuid_to_block_id)
            or (link.dst.node not in uuid_to_block_id)
        ):
            continue
        src_node_id = uuid_to_block_id[link.src.node]
        dst_node_id = uuid_to_block_id[link.dst.node]
        src_port_index = int(link.src.port)
        dst_port_index = int(link.dst.port)
        builder.connect(
            blocks[src_node_id].output_ports[src_port_index],
            blocks[dst_node_id].input_ports[dst_port_index],
        )

    return builder.build(name=name)


def simulation_settings(
    config: json_types.Configuration, recorded_signals: dict[str, CacheSource] = None
):
    max_interval_between_samples = config.max_interval_between_samples
    max_step = config.solver.max_step

    # HACK: use dto allow interpolation setting to control results
    # sampling rate by controlling solver step size. not great.
    if (
        max_interval_between_samples is not None
        and max_interval_between_samples > 0
        and max_interval_between_samples < max_step
    ):
        max_step = max_interval_between_samples
        logging.warning(
            "max_step_size reduced to %s to match max_interval_between_samples",
            max_step,
        )

    # FIXME: this is clearly a useless remapping. however, DASH-1412 requires that
    # model.json use "non-stiff" and "stiff", in which case this remapping is necessary.
    if config.solver.method in ["stiff", "default", "RK45"]:
        method = "default"
    elif config.solver.method in ["non-stiff", "Kvaerno5"]:
        method = "Kvaerno5"
    else:
        raise ValueError(
            f"Unrecognized value for solver method: {config.solver.method}"
        )

    ode_options = ODESolverOptions(
        rtol=config.solver.relative_tolerance,
        atol=config.solver.absolute_tolerance,
        max_step_size=max_step,
        min_step_size=config.solver.min_step,
        max_steps=config.max_minor_steps_per_major_step,
        method=method,
    )

    results_options = ResultsOptions(max_interval_between_samples)

    global_discrete_interval = config.sample_time
    if (
        results_options.max_interval_between_samples is not None
        and results_options.max_interval_between_samples > 0
    ):
        global_discrete_interval = min(
            global_discrete_interval, results_options.max_interval_between_samples
        )

    simulator_options = SimulatorOptions(
        max_major_steps=config.max_major_steps,
        max_major_step_length=global_discrete_interval,
        return_context=False,
        recorded_signals=recorded_signals,
    )

    logging.info("Simulation settings: %s", simulator_options)
    logging.info("ODE solver settings: %s", ode_options)
    logging.info("Results settings: %s", results_options)

    return ode_options, results_options, simulator_options


class AppInterface:
    def __init__(
        self,
        model: json_types.Model,
        datadir: str = None,
        logsdir: str = None,
        block_overrides: dict[str, SystemBase] = None,
    ):
        root_id = "root"
        self.context: ContextBase = None
        self.root_diagram = model.diagram
        model_parameters = eval_parameters(
            root_id,
            default_parameters=model.parameter_definitions,
            instance_parameters=model.parameters,
        )
        if model.configuration.workspace:
            # if it's not an empty dict, it should have ref to initscript
            # @am. this is not great but works for now.
            if model.configuration.workspace.init_scripts:
                filename = model.configuration.workspace.init_scripts[0]["file_name"]
                model_parameters = eval_init_script(root_id, filename, model_parameters)

        recorded_signals = {}
        self.diagram = make_subdiagram(
            root_id,
            self.root_diagram,
            model.subdiagrams,
            namespace_params=model_parameters,
            block_overrides=block_overrides,
            global_discrete_interval=model.configuration.sample_time,
            record_mode=model.configuration.record_mode,
            recorded_signals=recorded_signals,
        )
        (
            self.ode_options,
            self.results_options,
            self.simulator_options,
        ) = simulation_settings(model.configuration, recorded_signals=recorded_signals)
        self.configuration = model.configuration
        self.datadir = datadir
        self.logsdir = logsdir

    def simulate(
        self,
        t=None,
        write_binary_results=False,
        record_outputs=True,
        ode_options: ODESolverOptions = None,
        simulator_options: SimulatorOptions = None,
    ) -> dict[str, jnp.ndarray]:
        try:
            self.context = self.diagram.create_context()
            logging.debug("Context created")

            # Most type errors will be caught when the context is created above.
            # See lynx.framework.context_factory._check_types. However, it
            # won't catch errors when there is a mismatch between an input type and
            # the state in stateful blocks. The call to check_types does that.
            self.diagram.check_types(self.context)
        except Exception as exc:
            # try / catch here only to provide a breakpoint site for inspection
            # from within the debugger. This may also become the site where
            # chained exceptions are parsed and normalized.
            raise exc

        # Write 'signal_types.json'
        if self.logsdir is not None:
            os.makedirs(self.logsdir, exist_ok=True)
            context = self.context
            signal_types = []
            for node in self.root_diagram.nodes:
                block = self.diagram[node.name]
                for port_idx, out_port in enumerate(block.output_ports):
                    val = out_port.eval(context)
                    # jp's trick. 'val' may have a Python type (int or float) in which
                    # case it doesn't have a dtype. So turn it into an array and then
                    # retrive the dtype. While a Python type doesn't have a dtype,
                    # for some reason one can apply np.shape(..) to it. Go figure.
                    signal_type = _map_types(np.array(val).dtype, np.shape(val))
                    ui_desc = {
                        "path": node.name + "." + block.output_ports[port_idx].name,
                        "port_index": port_idx,
                        "cml_type": signal_type,
                    }
                    signal_types.append(ui_desc)

            signal_types_file = os.path.join(self.logsdir, "signal_types.json")
            with open(signal_types_file, "w", encoding="utf-8") as outfile:
                json.dump(signal_types, outfile, indent=2, sort_keys=False)

        if t is None:
            t = self.configuration.stop_time

        start_time = float(self.configuration.start_time)

        options = self.simulator_options
        if simulator_options is not None:
            options = simulator_options

        results = simulate(
            self.diagram,
            self.context,
            (start_time, float(t)),
            options=options,
            ode_options=ode_options if ode_options is not None else self.ode_options,
            results_options=self.results_options,
            recorded_signals=options.recorded_signals,
        )

        # Calls to model.simulate are expecting a dict of outputs including time.
        results.outputs["time"] = results.time

        if record_outputs:
            tf_res = results.outputs["time"][-1]
            if tf_res < t:
                print(
                    f"WARNING: Simulation ended at tf={tf_res} rather than requested {t=}.  "
                    "This is most likely due to reaching max_major_steps. "
                    "Try re-running simulation with larger max_major_steps. "
                )

        if write_binary_results:
            if self.datadir is None:
                raise Exception("cannnot write binary results when datadir is None")
            write_binary_results_f(results.outputs, self.datadir, self.logsdir)

        return results.outputs
