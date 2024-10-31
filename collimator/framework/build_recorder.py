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
This module records the build commands and generates the code for the diagram.
"""

from contextlib import contextmanager
import inspect
from typing import Any, Callable, NamedTuple, TYPE_CHECKING, Union

import jax
import numpy as np

if TYPE_CHECKING:
    from .diagram_builder import DiagramBuilder
    from .diagram import Diagram
    from .leaf_system import LeafSystem
    from .parameter import Parameter
    from .port import InputPort, OutputPort
    from .system_base import SystemBase
    from ..experimental.acausal.component_library.component_base import ComponentBase
    from ..library.reference_subdiagram import ReferenceSubdiagramProtocol


__IMPORTS__ = set()

__BUILD_CMDS__ = []

__BUILDERS_ID__ = {}

__BLOCK_TO_BUILDER__ = {}

__REF_SUBDIAGRAMS__ = {}  # ref_id to variable name

__IS_RECORDING__ = False


def _init_kwargs_to_str(block, kwargs):
    kwargs_str = {}
    for name, value in kwargs.items():
        if name in block.parameters:
            p = block.parameters[name]
            if not p.is_python_expr and isinstance(p.value, str):
                kwargs_str[name] = repr(p.value)
            else:
                kwargs_str[name] = str(p)
        else:
            kwargs_str[name] = str(value)
    return kwargs_str


def _get_builder_name(builder: "DiagramBuilder"):
    return f"{__BUILDERS_ID__[builder]}_builder"


def _get_diagram_name(builder: "DiagramBuilder"):
    return __BUILDERS_ID__[builder]


def _get_block_name(block: "SystemBase"):
    if block in __BLOCK_TO_BUILDER__:
        builder = __BLOCK_TO_BUILDER__[block]
        return f"{_get_diagram_name(builder)}_{block.name}"
    return block.name


def _unindent(code: str):
    lines = code.split("\n")
    prefix_spaces = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[prefix_spaces:] for line in lines)


class CreateAcausalDiagramCommand(NamedTuple):
    def __str__(self):
        # FIXME: should parse the __init__ args but for the purpose of showing
        # code that works in the UI, that will do.
        return "ad = AcausalDiagram()"


class CreateAcausalCompilerCommand(NamedTuple):
    def __str__(self):
        # FIXME: should parse the __init__ args but for the purpose of showing
        # code that works in the UI, that will do.
        return "ac = AcausalCompiler(ee, ad)"


class CompileAcausalDiagramCommand(NamedTuple):
    system: "LeafSystem"

    def __str__(self):
        return f"{_get_block_name(self.system)} = ac.generate_acausal_system(name='{self.system.name}')"


class CreateEqnEnvCommand(NamedTuple):
    def __str__(self):
        return "ee = EqnEnv()"


class ConnectAcausalPortsCommand(NamedTuple):
    block1: "ComponentBase"
    port1: str
    block2: "ComponentBase"
    port2: str

    def __str__(self):
        args = [
            self.block1.name,
            repr(self.port1),
            self.block2.name,
            repr(self.port2),
        ]
        args = ", ".join(args)
        return f"ad.connect({args})"


class CreateAcausalBlockCommand(NamedTuple):
    block: "ComponentBase"
    init_fn: Callable
    args: list[Any]
    kwargs: dict[str, Any]

    def __str__(self):
        args = [
            repr(v) if isinstance(v, str) else str(v)
            for v in self.args[1:]  # skip EqnEnv
        ]
        args = ["ee"] + args
        args_str = ", ".join(args)
        kwargs_str = ""

        if self.kwargs:
            kwargs_str = ", ".join(
                f"{k}={repr(v)}" if isinstance(v, str) else f"{k}={v}"
                for k, v in self.kwargs.items()
            )

        all_args = ", ".join(x for x in [args_str, kwargs_str] if x)
        block_name = _get_block_name(self.block)

        domain = self.block.__class__.__module__.split(".")[-1]
        clazz = f"{domain}.{self.block.__class__.__qualname__}"
        return f"{block_name} = {clazz}({all_args})"


class CreateBlockCommand(NamedTuple):
    block: "LeafSystem"
    init_fn: Callable
    args: list[Any]
    kwargs: dict[str, Any]

    def __str__(self):
        args = [repr(v) if isinstance(v, str) else str(v) for v in self.args]
        args_str = ", ".join(args)

        kwargs_str = ""
        if self.kwargs:
            kwargs_str = _init_kwargs_to_str(self.block, self.kwargs)
            kwargs_str = ", ".join(
                f"{k}={v}" for k, v in kwargs_str.items() if k not in args
            )

        all_args = ", ".join(x for x in [args_str, kwargs_str] if x)
        block_name = _get_block_name(self.block)

        clazz = f"library.{self.block.__class__.__name__}"
        return f"{block_name} = {clazz}({all_args})"


class CreateParameterCommand(NamedTuple):
    args: dict[str, Any]

    def __str__(self):
        name = self.args.get("name")
        value = self.args.get("value")
        return f'{name} = Parameter(name="{name}", value={value})'


class AddBlockCommand(NamedTuple):
    builder: "DiagramBuilder"
    blocks: list["SystemBase"]

    def __str__(self):
        blocks_str = ", ".join(_get_block_name(b) for b in self.blocks)
        builder_name = _get_builder_name(self.builder)

        return f"{builder_name}.add({blocks_str})"


class ConnectPortsCommand(NamedTuple):
    builder: "DiagramBuilder"
    outport: "OutputPort"
    inport: "InputPort"

    def __str__(self):
        builder_name = _get_builder_name(self.builder)
        oport_block = _get_block_name(self.outport.system)
        iport_block = _get_block_name(self.inport.system)
        oport_str = f"{oport_block}.output_ports[{self.outport.index}]"
        iport_str = f"{iport_block}.input_ports[{self.inport.index}]"
        return f"{builder_name}.connect({oport_str}, {iport_str})"


class ExportPortCommand(NamedTuple):
    builder: "DiagramBuilder"
    block: "SystemBase"
    side: str  # input or output
    port: int
    port_name: str

    def __str__(self):
        builder_name = _get_builder_name(self.builder)
        fn_name = f"{builder_name}.export_{self.side}"
        block_name = _get_block_name(self.block)
        args = f'{block_name}.{self.side}_ports[{self.port}], "{self.port_name}"'
        return f"{fn_name}({args})"


class BuildDiagramCommand(NamedTuple):
    builder: "DiagramBuilder"
    diagram: "Diagram"
    parameters: dict[str, "Parameter"]

    def __str__(self):
        builder_name = _get_builder_name(self.builder)
        diagram_name = _get_block_name(self.diagram)

        if self.parameters:
            params_str = ", ".join(f'"{p}": {p}' for p in self.parameters)
            params_str = f"{{{params_str}}}"
            return f'{diagram_name} = {builder_name}.build("{self.diagram.name}", parameters={params_str})'
        return f'{diagram_name} = {builder_name}.build("{self.diagram.name}")'


class CreateRefSubdiagramCommand(NamedTuple):
    diagram: "Diagram"
    ref_id: str
    instance_name: str
    instance_parameters: dict[str, "Parameter"]

    def __str__(self):
        var_name = __REF_SUBDIAGRAMS__[self.ref_id]
        block_name = _get_block_name(self.diagram)
        if self.instance_parameters:
            params_str = ", ".join(f'"{p}": {p}' for p in self.instance_parameters)
            params_str = f"{{{params_str}}}"
            args = (
                f'{var_name}, "{self.instance_name}", instance_parameters={params_str}'
            )
            return f"{block_name} = library.ReferenceSubdiagram.create_diagram({args})"
        return f'{block_name} = library.ReferenceSubdiagram.create_diagram({var_name}, "{self.instance_name}")'


class RegisterRefSubdiagramCommand(NamedTuple):
    constructor: "ReferenceSubdiagramProtocol"
    parameter_definitions: list["Parameter"]
    ref_id: str

    def __str__(self):
        fn_source = inspect.getsource(self.constructor)
        fn_source = _unindent(fn_source)
        fn_name = self.constructor.__name__
        params = [
            f'Parameter(name="{p.name}", value={p.value})'
            for p in self.parameter_definitions
        ]
        params = ", ".join(params)
        args = f"{fn_name}, parameter_definitions=[{params}]"
        var_name = f"ref_id_{len(__REF_SUBDIAGRAMS__)}"
        __REF_SUBDIAGRAMS__[self.ref_id] = var_name
        return (
            f"\n{fn_source}\n{var_name} = library.ReferenceSubdiagram.register({args})"
        )


def clear():
    __IMPORTS__.clear()
    __BUILD_CMDS__.clear()
    __BUILDERS_ID__.clear()
    __BLOCK_TO_BUILDER__.clear()
    __REF_SUBDIAGRAMS__.clear()


@contextmanager
def paused():
    global __IS_RECORDING__
    was_recording = __IS_RECORDING__
    __IS_RECORDING__ = False

    yield

    if was_recording:
        resume()


def pause():
    global __IS_RECORDING__
    __IS_RECORDING__ = False


def resume():
    global __IS_RECORDING__
    __IS_RECORDING__ = True


def start():
    clear()
    resume()


def stop():
    pause()
    clear()


def is_recording():
    return __IS_RECORDING__


def _action(func):
    def wrapper(*args, **kwargs):
        if __IS_RECORDING__:
            return func(*args, **kwargs)

    return wrapper


def _is_default(value, default):
    from .parameter import Parameter  # avoid circular import

    if isinstance(value, Parameter):
        return False
    if default is inspect.Parameter.empty:
        return False
    array_types = (jax.Array, np.ndarray)
    if not isinstance(value, array_types) and not isinstance(default, array_types):
        return value == default
    if isinstance(value, array_types) and isinstance(default, array_types):
        return np.array_equal(value, default)
    return False


def _add_acausal_domain_to_imports(full_mod_path: str):
    domain = full_mod_path.split(".")[-1]
    __IMPORTS__.add(f"from collimator.experimental import {domain}")


@_action
def create_block(block: Union["LeafSystem", "ComponentBase"], init_fn, *args, **kwargs):
    sig = inspect.signature(init_fn)
    # Bind the arguments to the signature
    bound_args = sig.bind(block, *args, **kwargs)
    bound_args.apply_defaults()

    # Filter out keyword arguments set to their default values
    filtered_kwargs = {
        k: v
        for k, v in bound_args.arguments.items()
        if k in kwargs and not _is_default(v, sig.parameters[k].default)
    }

    for i, cmd in enumerate(__BUILD_CMDS__):
        if isinstance(cmd, CreateBlockCommand) and cmd.block == block:
            # replace the command with the new one
            __BUILD_CMDS__[i] = CreateBlockCommand(
                block, init_fn, args, filtered_kwargs
            )
            return
        elif isinstance(cmd, CreateAcausalBlockCommand) and cmd.block == block:
            # replace the command with the new one
            __BUILD_CMDS__[i] = CreateAcausalBlockCommand(
                block, init_fn, args, filtered_kwargs
            )
            return

    from .leaf_system import LeafSystem  # avoid circular import

    if isinstance(block, LeafSystem):
        __IMPORTS__.add("from collimator import library")
        __BUILD_CMDS__.append(CreateBlockCommand(block, init_fn, args, filtered_kwargs))
    else:
        _add_acausal_domain_to_imports(block.__class__.__module__)
        __BUILD_CMDS__.append(
            CreateAcausalBlockCommand(block, init_fn, args, filtered_kwargs)
        )


@_action
def create_acausal_compiler():
    __IMPORTS__.add("from collimator.experimental import AcausalCompiler")
    __BUILD_CMDS__.append(CreateAcausalCompilerCommand())


@_action
def compile_acausal_diagram(system: "LeafSystem"):
    __BUILD_CMDS__.append(CompileAcausalDiagramCommand(system))


@_action
def create_eqn_env():
    __IMPORTS__.add("from collimator.experimental import EqnEnv")
    __BUILD_CMDS__.append(CreateEqnEnvCommand())


@_action
def create_acausal_diagram():
    __IMPORTS__.add("from collimator.experimental import AcausalDiagram")
    __BUILD_CMDS__.append(CreateAcausalDiagramCommand())


@_action
def connect_acausal_ports(
    block1: "ComponentBase", port1: str, block2: "ComponentBase", port2: str
):
    __BUILD_CMDS__.append(ConnectAcausalPortsCommand(block1, port1, block2, port2))


@_action
def create_parameter(args):
    if args.get("is_python_expr", False):
        args.pop("py_namespace")
    if args.get("name", False):
        __IMPORTS__.add("from collimator.framework import Parameter")
        __BUILD_CMDS__.append(CreateParameterCommand(args))


@_action
def add_block(builder: "DiagramBuilder", blocks: list["SystemBase"]):
    for block in blocks:
        __BLOCK_TO_BUILDER__[block] = builder
    __BUILD_CMDS__.append(AddBlockCommand(builder, blocks))


@_action
def connect_ports(builder: "DiagramBuilder", src: "OutputPort", dst: "InputPort"):
    __BUILD_CMDS__.append(ConnectPortsCommand(builder, src, dst))


@_action
def export_port(
    builder: "DiagramBuilder", block_name: str, side: str, port: int, port_name: str
):
    __BUILD_CMDS__.append(ExportPortCommand(builder, block_name, side, port, port_name))


@_action
def build_diagram(
    builder: "DiagramBuilder", diagram: "Diagram", parameters: dict[str, "Parameter"]
):
    __BUILDERS_ID__[builder] = diagram.name
    __BUILD_CMDS__.append(BuildDiagramCommand(builder, diagram, parameters))


@_action
def create_ref_subdiagram(
    diagram: "Diagram",
    ref_id: str,
    instance_name: str,
    instance_parameters: dict[str, Any],
):
    # FIXME: this is not used because it wouldn't work for complex cases like
    # how the model.json is loaded in from_model_json. Instead reference submodels
    # will be generated as groups.
    __BUILD_CMDS__.append(
        CreateRefSubdiagramCommand(diagram, ref_id, instance_name, instance_parameters)
    )


@_action
def register_ref_subdiagram(
    constructor: "ReferenceSubdiagramProtocol",
    parameter_definitions: list["Parameter"],
    ref_id: str,
):
    # FIXME: this is not used because it wouldn't work for complex cases like
    # how the model.json is loaded in from_model_json. Instead reference submodels
    # will be generated as groups.
    __BUILD_CMDS__.append(
        RegisterRefSubdiagramCommand(constructor, parameter_definitions, ref_id)
    )


def get_commands():
    return __BUILD_CMDS__


def get_diagram_builders() -> list[str]:
    return [
        f"{_get_builder_name(builder)} = DiagramBuilder()"
        for builder in __BUILDERS_ID__
    ]


def generate_code():
    imports = ["from collimator import DiagramBuilder"] + list(__IMPORTS__)
    code = sorted(imports)
    code.append("")
    code.extend(c for c in get_diagram_builders())
    code.append("")
    code.extend(str(c) for c in get_commands())
    return "\n".join(code)
