"""This file contains conversion logic from wildcat to collimator model.json.
Collimator model.json to wildcat conversion is done in model_interface.py."""

from uuid import uuid4

from lynx.cli.types import (
    Configuration,
    Diagram,
    Link,
    Model,
    Node,
    Parameter,
    Port,
    Reference,
    Subdiagrams,
)
from lynx.framework import (
    CacheSource,
    Diagram as LynxDiagram,
    SystemBase,
)
from lynx.library import (
    CustomJaxBlock,
    CustomPythonBlock,
    Demultiplexer,
    Derivative,
    DiscreteClock,
    DiscretePID,
    IOPort,
    LTISystem,
    Multiplexer,
    PID,
    ReduceBlock,
    ReferenceSubdiagram,
    Sine,
    TransferFunction,
)

_CL_LIBRARY = {
    "Abs",
    "Adder",
    "Chirp",
    "Clock",
    "Comparator",
    "Constant",
    "CosineWave",
    "CrossProduct",
    "DataSource",
    "DeadZone",
    "Demux",
    "Derivative",
    "DerivativeDiscrete",
    "DotProduct",
    "EdgeDetection",
    "Exponent",
    "Gain",
    "IfThenElse",
    "Inport",
    "Integrator",
    "IntegratorDiscrete",
    "LogicalOperator",
    "LookupTable1d",
    "LookupTable2d",
    "Log",
    "MatrixInversion",
    "MatrixMultiplication",
    "MatrixTransposition",
    "MinMax",
    "ModelicaFMU",
    "Mux",
    "Offset",
    "Outport",
    "PID",
    "PID_Discrete",
    "Power",
    "Predictor",
    "Product",
    "ProductOfElements",
    "Pulse",
    "PythonScript",
    "Quantizer",
    "Ramp",
    "Reciprocal",
    "Saturate",
    "Sawtooth",
    "ScalarBroadcast",
    "SineWave",
    "SINDy",
    "Slice",
    "SquareRoot",
    "Stack",
    "StateSpace",
    "Step",
    "SumOfElements",
    "TransferFunction",
    "Trigonometric",
    "UnitDelay",
    "ZeroOrderHold",
}

# Map of wildcat block param names to collimator block param names.
_PARAM_NAME_MAP = {
    "Integrator": {
        "initial_continuous_state": "initial_states",
    },
    "IntegratorDiscrete": {
        "initial_state": "initial_states",
    },
    "DiscretePID": {
        "kp": "Kp",
        "ki": "Ki",
        "kd": "Kd",
    },
    "LTISystem": {
        "initialize_states": "initial_states",
    },
}


class IOPortExportError(Exception):
    ...


def _param_name(block, param_name):
    block_cls = block.__class__.__name__
    if block_cls in _PARAM_NAME_MAP:
        if param_name in _PARAM_NAME_MAP[block_cls]:
            return _PARAM_NAME_MAP[block_cls][param_name]
    return param_name


def _wc_to_cl_parameters(block: SystemBase) -> dict[str, Parameter]:
    """Parse parameters of a wildcat block to a Collimator-compatible format"""

    # Following blocks are instances of LTISystem so we want to ignore A, B, C, D
    if isinstance(block, PID):
        kp = block.instance_parameters["kp"].string_value
        ki = block.instance_parameters["ki"].string_value
        kd = block.instance_parameters["kd"].string_value
        n = block.instance_parameters["n"].string_value
        initial_state = block.instance_parameters["initial_state"].string_value
        return {
            "Kp": Parameter(value=kp),
            "Ki": Parameter(value=ki),
            "Kd": Parameter(value=kd),
            "N": Parameter(value=n),
            "initial_state": Parameter(value=initial_state),
        }
    elif isinstance(block, TransferFunction):
        num = block.instance_parameters["num"].string_value
        den = block.instance_parameters["den"].string_value
        return {
            "numerator_coefficients": Parameter(value=num),
            "denominator_coefficients": Parameter(value=den),
        }
    elif isinstance(block, Derivative):
        n = block.instance_parameters["N"].string_value
        return {
            "N": Parameter(value=n),
        }

    params = {}
    params.update(
        {
            _param_name(block, k): Parameter(value=p.string_value)
            for k, p in block.instance_parameters.items()
        }
    )

    for k, v in block.instance_parameters.items():
        if v is None:
            continue
        is_string = isinstance(v.evaluated_value, str)
        params[_param_name(block, k)] = Parameter(
            value=v.string_value, is_string=is_string
        )

    return params


def _wc_to_cl_ports(ports: list[CacheSource], kind, params: dict = None) -> list[dict]:
    """Parse input ports of a Lynx block to a Collimator-compatible format"""
    if params is None:
        params = {}
    return [
        {
            "name": port.name,
            "kind": kind,
            "parameters": params[port.name] if port.name in params else {},
        }
        for port in ports
    ]


def _wc_to_cl_iports(node: SystemBase) -> list[dict]:
    """Parse input ports of a Lynx block to a Collimator-compatible format"""
    kind = "dynamic" if isinstance(node, ReduceBlock) else "static"
    return _wc_to_cl_ports(node.input_ports, kind)


def _wc_to_cl_oports(node: SystemBase) -> list[dict]:
    """Parse input ports of a Lynx block to a Collimator-compatible format"""
    params = {}
    if isinstance(node, CustomJaxBlock):
        for port_name, port_param in node.output_port_params.items():
            params[port_name] = {
                "dtype": Parameter(value=str(port_param["dtype"]), is_string=True),
                "shape": Parameter(value=str(port_param["shape"])),
            }
    kind = "dynamic" if isinstance(node, Demultiplexer) else "static"
    return _wc_to_cl_ports(node.output_ports, kind, params=params)


def _get_block_type(node: SystemBase) -> str:
    if isinstance(node, Sine):
        return "core.SineWave"
    elif isinstance(node, Multiplexer):
        return "core.Mux"
    elif isinstance(node, Demultiplexer):
        return "core.Demux"
    elif isinstance(node, DiscretePID):
        return "core.PID_Discrete"
    elif isinstance(node, DiscreteClock):
        return "core.Clock"
    elif type(node) is LTISystem:
        return "core.StateSpace"
    elif isinstance(node, LynxDiagram):
        if node.ref_id is not None:
            return "core.ReferenceSubmodel"
        else:
            return "core.Group"
    elif isinstance(node, CustomPythonBlock):
        return "core.PythonScript"
    elif node.__class__.__name__ in _CL_LIBRARY:
        return f"core.{node.__class__.__name__}"

    return None


def _get_time_mode(node: SystemBase) -> str:
    _DISCRETE_BLOCKS = {DiscretePID, DiscreteClock}
    if any(isinstance(node, clazz) for clazz in _DISCRETE_BLOCKS):
        return "discrete"

    return "continuous"


def _get_ref_submodel_uuid(node: LynxDiagram) -> str:
    if not isinstance(node, LynxDiagram):
        return None
    return node.ref_id


def _wc_to_cl_block(node: SystemBase) -> Node:
    block_type = _get_block_type(node)

    if block_type is None:
        raise NotImplementedError(
            f"Block type {node.__class__.__name__} is not supported"
        )

    return Node(
        name=node.name,
        type=_get_block_type(node),
        inputs=_wc_to_cl_iports(node),
        outputs=_wc_to_cl_oports(node),
        parameters=_wc_to_cl_parameters(node),
        uuid=str(uuid4()),
        submodel_reference_uuid=_get_ref_submodel_uuid(node),
        time_mode=_get_time_mode(node),
    )


def _wc_to_cl_links(
    diagram: LynxDiagram, nodes: dict[SystemBase, Node]
) -> list[dict[Link]]:
    links = []
    for iport, oport in diagram.connection_map.items():
        input_sys, input_idx = iport
        output_sys, output_idx = oport
        links.append(
            Link(
                uuid=str(uuid4()),
                src=Port(node=nodes[output_sys].uuid, port=output_idx),
                dst=Port(node=nodes[input_sys].uuid, port=input_idx),
            )
        )
    return links


def _make_inport(name: str) -> Node:
    return Node(
        name=name,
        type="core.Inport",
        outputs=[
            {
                "name": "out_0",
                "kind": "static",
                "parameters": {},
            }
        ],
        parameters={
            "description": Parameter(is_string=True, value=""),
            "port_id": Parameter(value="0"),
        },
        uuid=str(uuid4()),
    )


def _make_outport(name: str) -> Node:
    return Node(
        name=name,
        type="core.Outport",
        inputs=[
            {
                "name": "in_0",
                "kind": "static",
                "parameters": {},
            }
        ],
        parameters={
            "description": Parameter(is_string=True, value=""),
            "port_id": Parameter(value="0"),
        },
        uuid=str(uuid4()),
    )


def _make_inport_and_link(name: str, node_uuid: str, idx: int) -> tuple[Node, Link]:
    inport = _make_inport(name)
    link = Link(
        uuid=str(uuid4()),
        src=Port(node=inport.uuid, port=0),
        dst=Port(node=node_uuid, port=idx),
    )
    return inport, link


def _make_outport_and_link(name: str, node_uuid: str, idx: int) -> tuple[Node, Link]:
    outport = _make_outport(name)
    link = Link(
        uuid=str(uuid4()),
        src=Port(node=node_uuid, port=idx),
        dst=Port(node=outport.uuid, port=0),
    )
    return outport, link


def _check_ioports_export_error(diagram: LynxDiagram) -> bool:
    for node in diagram:
        if isinstance(node, IOPort):
            loc = (node, 0)
            is_input_port = loc in diagram.exported_input_ports
            is_output_port = loc in diagram.exported_output_ports

            if not is_input_port and not is_output_port:
                raise IOPortExportError(
                    f"IOPort {node.name} is not exported as input or output"
                )
            elif is_input_port and is_output_port:
                raise IOPortExportError(
                    f"IOPort {node.name} is exported as both input and output"
                )


RefId = str


def _wc_to_cl_diagram(diagram: LynxDiagram) -> tuple[Model, dict[RefId, Model]]:
    _check_ioports_export_error(diagram)
    groups = {node for node in diagram if isinstance(node, LynxDiagram)}

    # TODO: check that the diagrams corresponding to the same submodel references
    # are the same

    # Convert all non IOPort nodes
    nodes = {
        node: _wc_to_cl_block(node) for node in diagram if not isinstance(node, IOPort)
    }

    ioports_links = []
    ioports = []
    # create Inports and Outports for exported ports
    for node, idx in diagram.exported_input_ports:
        if isinstance(node, IOPort):
            nodes[node] = _make_inport(node.name)
        else:
            inport, link = _make_inport_and_link(
                f"{node.name}_inport_{idx}", nodes[node].uuid, idx
            )
            ioports.append(inport)
            ioports_links.append(link)

    for node, idx in diagram.exported_output_ports:
        if isinstance(node, IOPort):
            nodes[node] = _make_outport(node.name)
        else:
            outport, link = _make_outport_and_link(
                f"{node.name}_outport_{idx}", nodes[node].uuid, idx
            )
            ioports.append(outport)
            ioports_links.append(link)

    # process reference submodels
    ref_subdiagrams = {}
    for node in diagram:
        if (
            not isinstance(node, LynxDiagram)
            or node.ref_id is None
            or node.ref_id in ref_subdiagrams
        ):
            continue
        ref_subdiagram, ref_ref_subdiagrams = _wc_to_cl_diagram(node)
        ref_subdiagrams.update(ref_ref_subdiagrams)
        ref_subdiagrams[node.ref_id] = ref_subdiagram

    subdiagrams = {}
    references = {}
    for group in groups:
        subdiagram, ref_ref_subdiagrams = _wc_to_cl_diagram(group)
        ref_subdiagrams.update(ref_ref_subdiagrams)
        references[nodes[group].uuid] = Reference(diagram_uuid=subdiagram.diagram.uuid)
        subdiagrams[subdiagram.diagram.uuid] = subdiagram.diagram
        subdiagrams.update(subdiagram.subdiagrams.diagrams)
        references.update(subdiagram.subdiagrams.references)

    root_diagram = Diagram(
        uuid=str(uuid4()),
        links=_wc_to_cl_links(diagram, nodes) + ioports_links,
        nodes=list(nodes.values()) + ioports,
        annotations=None,
    )
    parameter_definitions = None
    if diagram.ref_id is not None:
        parameter_definitions = ReferenceSubdiagram.get_parameter_definitions(
            diagram.ref_id
        )
    parameters = None
    if diagram.instance_parameters is not None:
        parameters = {}
        for k, v in diagram.instance_parameters.items():
            is_string = isinstance(v.evaluated_value, str)
            parameters[k] = Parameter(value=v.string_value, is_string=is_string)

    return (
        Model(
            uuid=diagram.system_id,
            name=diagram.name,
            diagram=root_diagram,
            subdiagrams=Subdiagrams(
                diagrams=subdiagrams,
                references=references,
            ),
            parameters=parameters,
            parameter_definitions=parameter_definitions,
        ),
        ref_subdiagrams,
    )


def convert(
    wc_diagram: LynxDiagram,
    configuration: Configuration = None,
) -> tuple[Model, dict[str, Model]]:
    """Convert a Lynx diagram to a Collimator model.json."""
    model, reference_submodels = _wc_to_cl_diagram(wc_diagram)
    if configuration is not None:
        model.configuration = configuration
    return model, reference_submodels
