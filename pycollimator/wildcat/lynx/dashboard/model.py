import io
import json
import logging

from lynx.cli import types as cl_types
from lynx.cli import wc_to_cl_converter
from lynx.framework import Diagram
from lynx.library import ReferenceSubdiagram

from . import api


logger = logging.getLogger(__name__)


def get_reference_submodel(project_uuid: str, model_uuid: str) -> cl_types.Model:
    try:
        model_dict = api.call(f"/project/{project_uuid}/submodels/{model_uuid}", "GET")
    except api.CollimatorNotFoundError:
        return None
    model_json = json.dumps(model_dict)
    return cl_types.Model.from_json(io.StringIO(model_json))


def create_reference_submodel(
    project_uuid: str,
    model_uuid: str,
    model: cl_types.Model,
    parameter_definitions=list[cl_types.ParameterDefinition],
):
    return api.call(
        f"/project/{project_uuid}/submodels",
        "POST",
        body={
            "uuid": model_uuid,
            "name": model.name,
            "diagram": model.diagram,
            "parameter_definitions": parameter_definitions,
            "submodels": model.subdiagrams,
        },
    )


def update_reference_submodel(
    project_uuid: str,
    model_uuid: str,
    model: cl_types.Model,
    edit_id: str,
    parameter_definitions: list[cl_types.ParameterDefinition] = None,
):
    body = {
        "name": model.name,
        "diagram": model.diagram,
        "submodels": model.subdiagrams,
        "edit_id": edit_id,
    }

    if parameter_definitions is not None:
        body["parameter_definitions"] = parameter_definitions

    return api.call(
        f"/project/{project_uuid}/submodels/{model_uuid}",
        "PUT",
        body=body,
    )


def get_model(model_uuid: str) -> cl_types.Model:
    try:
        model_dict = api.call(f"/models/{model_uuid}", "GET")
    except api.CollimatorNotFoundError:
        return None
    model_json = json.dumps(model_dict)
    return cl_types.Model.from_json(io.StringIO(model_json))


def create_model(project_uuid: str, model: cl_types.Model):
    body = {
        "name": model.name,
        "diagram": model.diagram,
        "project_uuid": project_uuid,
        "configuration": model.configuration,
    }

    return api.call(
        "/models",
        "POST",
        body=body,
    )


def update_model(model_uuid: str, model: cl_types.Model):
    body = {
        "configuration": model.configuration,
        "diagram": model.diagram,
        "name": model.name,
        "submodels": model.subdiagrams,
        "version": model.version if model else 1,
    }

    if model.parameters is not None:
        body["parameters"] = model.parameters

    return api.call(
        f"/models/{model_uuid}",
        "PUT",
        body=body,
    )


BlockNamePortIdPair = tuple[str, int]


def _find_link(
    diagram: cl_types.Diagram, src: BlockNamePortIdPair, dst: BlockNamePortIdPair
):
    src_block = _find_node_by_name(diagram, src[0])
    dst_block = _find_node_by_name(diagram, dst[0])
    if not src_block or not dst_block:
        return None

    for link in diagram.links:
        node_match = src_block.uuid == link.src.node and dst_block.uuid == link.dst.node
        port_match = src[1] == link.src.port and dst[1] == link.dst.port
        if node_match and port_match:
            return link


def _find_node_by_name(diagram: cl_types.Diagram, name: str):
    for node in diagram.nodes:
        if node.name == name:
            return node


def _copy_uiprops(model1: cl_types.Model, model2: cl_types.Model):
    for link1 in model1.diagram.links:
        if link1.uiprops is None:
            continue

        src_node = model1.diagram.find_node(link1.src.node)
        dst_node = model1.diagram.find_node(link1.dst.node)
        link2 = _find_link(
            model2.diagram,
            (src_node.name, link1.src.port),
            (dst_node.name, link1.dst.port),
        )
        if link2 is not None:
            link2.uiprops = link1.uiprops

    for node1 in model1.diagram.nodes:
        if node1.uiprops is None:
            continue
        node2 = _find_node_by_name(model2.diagram, node1.name)
        if node2 is not None:
            node2.uiprops = node1.uiprops

    for group_uuid1, group_diagram1 in model1.subdiagrams.diagrams.items():
        group_diagram2 = model2.subdiagrams.diagrams.get(group_uuid1)
        if group_diagram2 is not None:
            _copy_uiprops(group_diagram1, group_diagram2)


def put_model(
    project_uuid: str,
    diagram: Diagram,
    configuration: cl_types.Configuration = None,
):
    model_json, ref_submodels = wc_to_cl_converter.convert(
        diagram,
        configuration=configuration,
    )

    for ref_id, ref_submodel in ref_submodels.items():
        submodel = get_reference_submodel(project_uuid, ref_id)
        submodel_uuid = ref_id
        param_def = ReferenceSubdiagram.get_parameter_definitions(ref_id)
        if submodel is None:
            logger.info("Creating submodel %s (%s)", ref_submodel.name, ref_id)
            response = create_reference_submodel(
                project_uuid, ref_id, ref_submodel, param_def
            )
            submodel_uuid = response["uuid"]
            edit_id = response["edit_id"]
        else:
            edit_id = submodel.edit_id
            _copy_uiprops(submodel, ref_submodel)
        logger.info("Updating submodel %s (%s)", ref_submodel.name, submodel_uuid)
        update_reference_submodel(
            project_uuid,
            submodel_uuid,
            ref_submodel,
            edit_id,
            parameter_definitions=param_def,
        )

    model_uuid = diagram.system_id
    model = get_model(model_uuid)
    if model is None:
        logger.info("Creating model %s", diagram.system_id)
        response = create_model(project_uuid, model_json)
        model_uuid = response["uuid"]
    else:
        model_json.version = model.version
        _copy_uiprops(model, model_json)

    logger.info("Updating model %s", diagram.system_id)

    update_model(model_uuid, model_json)

    return model_uuid
