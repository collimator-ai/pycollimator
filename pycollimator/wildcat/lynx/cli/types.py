"""
This module contains types that map directly to Collimator's JSON model format.

TODO: move this file to collimator submodule
"""

import dataclasses
import json
from typing import TextIO
from uuid import UUID, uuid4

from lynx.simulation import ODESolverOptions, ResultsOptions


@dataclasses.dataclass
class JSONObject:
    def to_json(self) -> str:
        obj_dict = dataclasses.asdict(self)
        return json.dumps(obj_dict, indent=2)


@dataclasses.dataclass
class Port:
    node: str
    port: int


@dataclasses.dataclass
class Link:
    uuid: str
    src: Port
    dst: Port
    uiprops: dict = None

    @staticmethod
    def from_dict(data: dict):
        return Link(
            uuid=data["uuid"],
            src=Port(**data["src"]) if "src" in data else None,
            dst=Port(**data["dst"]) if "dst" in data else None,
            uiprops=data.get("uiprops", {}),
        )


@dataclasses.dataclass
class Parameter(JSONObject):
    value: str
    is_string: bool = False

    @staticmethod
    def from_dict(data: dict):
        return Parameter(value=data["value"], is_string=data.get("is_string", False))


@dataclasses.dataclass
class IOPort:
    name: str
    kind: str  # static or dynamic
    parameters: dict[str, Parameter] = None
    record: bool = True

    @staticmethod
    def from_dict(data: dict):
        parameters = data.get("parameters", {})
        return IOPort(
            name=data["name"],
            kind=data.get("kind", "static"),
            parameters={k: Parameter.from_dict(v) for k, v in parameters.items()},
            record=data.get("record", False),
        )


@dataclasses.dataclass
class Node:
    name: str
    type: str = None
    inputs: list[IOPort] = dataclasses.field(default_factory=list)
    outputs: list[IOPort] = dataclasses.field(default_factory=list)
    parameters: dict[str, Parameter] = None
    uuid: str = None
    submodel_reference_uuid: str = None
    time_mode: str = None
    uiprops: dict = None

    @staticmethod
    def from_dict(data: dict):
        parameters = data.get("parameters", {})
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        return Node(
            uuid=data["uuid"],
            name=data["name"],
            type=data["type"],
            inputs=[IOPort.from_dict(d) for d in inputs] if inputs else [],
            outputs=[IOPort.from_dict(d) for d in outputs] if outputs else [],
            parameters={k: Parameter.from_dict(v) for k, v in parameters.items()},
            submodel_reference_uuid=data.get("submodel_reference_uuid", None),
            time_mode=data.get("time_mode", "continuous"),
            uiprops=data.get("uiprops", {}),
        )


BlockNamePortIdPair = tuple[str, int]


@dataclasses.dataclass
class Diagram(JSONObject):
    uuid: str
    links: list[Link]
    nodes: list[Node]
    annotations: list[dict] = None

    @staticmethod
    def from_dict(data: dict):
        return Diagram(
            uuid=data.get("uuid", str(uuid4())),
            links=[Link.from_dict(link) for link in data["links"]],
            nodes=[Node.from_dict(n) for n in data["nodes"]],
            annotations=data.get("annotations"),
        )

    def find_node(self, uuid: str):
        for node in self.nodes:
            if node.uuid == uuid:
                return node


@dataclasses.dataclass
class Reference:
    diagram_uuid: UUID


@dataclasses.dataclass
class ParameterDefinition(JSONObject):
    name: str
    default_value: str
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid4()))


@dataclasses.dataclass
class WorkSpace:
    init_scripts: list[dict] = dataclasses.field(default_factory=list)

    @staticmethod
    def from_dict(data: dict):
        return WorkSpace(init_scripts=data.get("init_scripts", []))


@dataclasses.dataclass
class SolverConfig:
    # see model-configuration.schema.json
    max_minor_steps_per_major_step: float = 1e3
    max_step: float = 1e6
    min_step: float = 0
    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-6
    method: str = "default"


@dataclasses.dataclass
class Configuration(JSONObject):
    # see model-configuration.schema.json
    record_mode: str = "all"
    solver: SolverConfig = dataclasses.field(default_factory=SolverConfig)
    sample_time: float = 0.1
    start_time: float = 0.0
    stop_time: float = 10.0
    workspace: WorkSpace = dataclasses.field(default_factory=WorkSpace)

    max_major_steps: int = None
    data_points_min: int = 0
    max_interval_between_samples: int = 1
    max_minor_steps_per_major_step: int = None

    @staticmethod
    def from_dict(data: dict):
        solver = None
        if "solver" in data:
            solver = SolverConfig(
                absolute_tolerance=data["solver"]["absolute_tolerance"],
                max_minor_steps_per_major_step=data["solver"].get(
                    "max_minor_steps_per_major_step", 1e3
                ),
                relative_tolerance=data["solver"]["relative_tolerance"],
                min_step=data["solver"]["min_step"],
                max_step=data["solver"]["max_step"],
                method=data["solver"].get("method", "default"),
            )

        max_major_steps = None
        if "max_major_steps" in data and data["max_major_steps"]:
            max_major_steps = int(data["max_major_steps"])
        return Configuration(
            record_mode=data.get("record_mode", "selected"),
            solver=solver,
            sample_time=data.get("sample_time", None),
            start_time=data.get("start_time", 0.0),
            stop_time=data["stop_time"],
            workspace=WorkSpace.from_dict(data.get("workspace", {})),
            max_major_steps=max_major_steps,
            data_points_min=data.get("data_points_min", 0),
            max_interval_between_samples=data.get("max_interval_between_samples", 1),
            max_minor_steps_per_major_step=data.get(
                "max_minor_steps_per_major_step", None
            ),
        )

    @staticmethod
    def from_wildcat_config(
        ode_options: ODESolverOptions,
        results_options: ResultsOptions,
        stop_time: float = 10.0,
        sample_time: float = 0.1,
        workspace=None,
    ):
        return Configuration(
            record_mode="all",
            solver=SolverConfig(
                absolute_tolerance=ode_options.atol,
                max_minor_steps_per_major_step=ode_options.max_steps,
                relative_tolerance=ode_options.rtol,
                min_step=ode_options.min_step_size,
                max_step=ode_options.max_step_size,
            ),
            sample_time=sample_time,
            stop_time=stop_time,
            max_interval_between_samples=results_options.max_interval_between_samples,
            workspace=workspace,
        )


@dataclasses.dataclass
class Subdiagrams:
    diagrams: dict[UUID, Diagram]
    references: dict[UUID, Reference]

    def get_diagram(self, group_block_uuid: UUID) -> Diagram:
        diagram_uuid = self.references[group_block_uuid].diagram_uuid
        return self.diagrams[diagram_uuid]


@dataclasses.dataclass
class Model(JSONObject):
    # intended to load model.json and submodel-uuid.ver.json
    uuid: str
    diagram: Diagram
    subdiagrams: Subdiagrams
    parameters: dict[str, Parameter]
    parameter_definitions: list[ParameterDefinition]
    name: str
    configuration: Configuration = dataclasses.field(default_factory=Configuration)
    version: int = 1

    # only for reference submodels
    edit_id: str = None

    # TODO:
    # state_machines:
    # submodel_configuration

    @staticmethod
    def from_json(fp: TextIO) -> "Model":
        return json.load(fp, cls=_ModelDecoder)


class _ModelDecoder(json.JSONDecoder):
    def decode(self, s):
        data = super(_ModelDecoder, self).decode(s)

        diagrams = None
        name = "root"

        if "name" in data:
            name = data["name"]

        # different options to handle both simworkerpy and simworker(go) json content.
        subdiagram_field_name = None
        if "subdiagrams" in data:
            subdiagram_field_name = "subdiagrams"

        elif "submodels" in data:
            subdiagram_field_name = "submodels"

        if subdiagram_field_name is not None:
            subdiagrams = data[subdiagram_field_name]
            if "diagrams" in subdiagrams:
                diagrams = {
                    k: Diagram.from_dict(d) for k, d in subdiagrams["diagrams"].items()
                }

            if "references" in subdiagrams:
                references = {
                    k: Reference(**d) for k, d in subdiagrams["references"].items()
                }

        # model and submodel parameters definitions
        parameter_definitions = None
        parameters = None
        if "parameter_definitions" in data and data["parameter_definitions"]:
            parameter_definitions = [
                ParameterDefinition(name=p["name"], default_value=p["default_value"])
                for p in data["parameter_definitions"]
            ]
        elif "parameters" in data and data["parameters"]:
            # this case catches:
            #   1] model.json["paramaters"], i.e. top level model parameters
            #   2] submodel-uuid-ver.json["parameters"], i.e. simworkergo submodel parameters_definitions
            parameters = {
                name: Parameter(param["value"], is_string=param.get("is_string", False))
                for name, param in data["parameters"].items()
            }

        configuration = None
        if "configuration" in data and data["configuration"]:
            configuration = Configuration.from_dict(data["configuration"])

        return Model(
            uuid=data.get("uuid", str(uuid4())),
            diagram=Diagram.from_dict(data["diagram"]),
            subdiagrams=Subdiagrams(diagrams=diagrams, references=references),
            parameter_definitions=parameter_definitions,
            parameters=parameters,
            configuration=configuration,
            name=name,
            version=data.get("version", 1),
            edit_id=data.get("edit_id", None),
        )
