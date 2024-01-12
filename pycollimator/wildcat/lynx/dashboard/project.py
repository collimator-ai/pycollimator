import concurrent.futures
import dataclasses
import os
import logging
import mimetypes
import requests
import tempfile
from typing import IO, AnyStr

from lynx.cli import SimulationContext, loads_model, register_reference_submodel
from lynx.cli import types as cl_types
from lynx.dashboard.schemas import FileSummary, ModelKind, ModelSummary, ProjectSummary

from . import api, model as model_api


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Project:
    models: dict[str, SimulationContext]
    files: list[str]


def _get_reference_submodels(
    project_uuid: str, model: cl_types.Model, memo: set[str]
) -> dict[str, cl_types.Model]:
    """Finds submodels & HLBs referenced by a given model or submodel"""
    if model.uuid in memo:
        return {}
    memo.add(model.uuid)
    refs = {}
    for node in model.diagram.nodes:
        if node.type != "core.ReferenceSubmodel":
            continue
        if node.submodel_reference_uuid in memo:
            continue
        submodel = model_api.get_reference_submodel(
            project_uuid, node.submodel_reference_uuid
        )
        refs[node.submodel_reference_uuid] = submodel
        refs.update(_get_reference_submodels(project_uuid, submodel, memo))

    for diagram_uuid, diagram in model.subdiagrams.diagrams.items():
        if diagram.uuid in memo:
            continue
        group = cl_types.Model(
            uuid=diagram_uuid,
            diagram=diagram,
            subdiagrams=cl_types.Subdiagrams(diagrams={}, references={}),
            parameters={},
            parameter_definitions=[],
            configuration=None,
            name="",
        )
        refs.update(_get_reference_submodels(project_uuid, group, memo))

    return refs


def _parse_project_summary(project_summary: dict) -> ProjectSummary:
    models = []
    for model in project_summary.get("models", []):
        models.append(
            ModelSummary(
                uuid=model["uuid"],
                kind=ModelKind(model.get("kind", "Model")),
                name=model["name"],
            )
        )

    submodels = []
    for model in project_summary.get("reference_submodels", []):
        submodels.append(
            ModelSummary(
                uuid=model["uuid"],
                kind=ModelKind(model.get("kind", "Submodel")),
                name=model["name"],
            )
        )

    files = []
    for file in project_summary.get("files", []):
        files.append(
            FileSummary(
                uuid=file["uuid"],
                name=file["name"],
                status=file["status"],
            )
        )

    return ProjectSummary(
        uuid=project_summary["uuid"],
        title=project_summary["title"],
        models=models,
        files=files,
        reference_submodels=submodels,
    )


def _download_file(file: FileSummary, project_uuid: str, destination: str):
    response = api.call(f"/projects/{project_uuid}/files/{file.uuid}/download", "GET")
    logger.debug("Downloading %s to %s", response["download_link"], destination)
    resp = requests.get(response["download_link"], verify=False)
    if resp.status_code != 200:
        logger.error(
            "Failed to download data file %s from project %s", file.name, project_uuid
        )
        return
    with open(destination, "wb") as f:
        f.write(resp.content)


def get_project(project_uuid: str, project_dir=None) -> Project:
    logger.info("Downloading project %s...", project_uuid)

    response = api.call(f"/projects/{project_uuid}", "GET", retries=3)
    project_summary = _parse_project_summary(response)

    # download project files
    files = []

    if project_dir is None:
        project_dir = tempfile.mkdtemp()
    logger.info("Project dir: %s", project_dir)

    for file in project_summary.files:
        if file.status == "processing_completed":
            dst = os.path.join(project_dir, file.name)
            _download_file(file, project_uuid, dst)
            files.append(dst)
        else:
            logger.warning(
                "File %s is not ready to be downloaded (status: %s)",
                file.name,
                file.status,
            )

    # Must first register submodels
    visited = set()
    ref_submodels: dict[str, cl_types.Model] = {}
    for model_summary in project_summary.reference_submodels:
        submodel = model_api.get_reference_submodel(project_uuid, model_summary.uuid)
        ref_submodels.update(
            {
                model_summary.uuid: submodel,
                **_get_reference_submodels(project_uuid, submodel, visited),
            }
        )
    for model_summary in project_summary.models:
        model = model_api.get_model(model_summary.uuid)
        ref_submodels.update(_get_reference_submodels(project_uuid, model, visited))
    for submodel_uuid, submodel in ref_submodels.items():
        logger.info("Registering submodel %s", submodel.name)
        register_reference_submodel(submodel_uuid, submodel)

    # Load models
    models = {}
    _globals = {}
    for model in project_summary.models:
        if model.kind == ModelKind.MODEL:
            model_json = model_api.get_model(model.uuid)
            init_scripts = model_json.configuration.workspace.init_scripts
            if len(init_scripts) > 1:
                raise NotImplementedError("Only one init script is supported")
            elif len(init_scripts) == 1 and init_scripts[0]:
                init_script_path = os.path.join(
                    project_dir, init_scripts[0]["file_name"]
                )
                if os.path.exists(init_script_path):
                    logger.info("Evaluating %s", init_scripts[0])
                    with open(init_script_path, "r") as f:
                        import numpy as np

                        _globals = {**globals(), "np": np}
                        exec(f.read(), _globals)
            if model_json.diagram.nodes:
                models[model.name] = loads_model(
                    model_json.to_json(), namespace=_globals
                )

    return Project(models=models, files=files)


def create_project(name: str) -> ProjectSummary:
    logger.info("Creating project %s...", name)
    response = api.call("/projects", "POST", body={"title": name})
    return _parse_project_summary(response)


def upload_file(project_uuid: str, name: str, fp: IO[AnyStr], overwrite=True):
    mime_type, _ = mimetypes.guess_type(name)
    size = os.fstat(fp.fileno()).st_size

    logger.info(
        "Uploading file %s (type: %s, size: %d) to project %s...",
        name,
        mime_type,
        size,
        project_uuid,
    )
    body = {
        "name": name,
        "content_type": mime_type,
        "overwrite": overwrite,
        "size": size,
    }
    put_url_response = api.call(f"/projects/{project_uuid}/files", "POST", body=body)
    s3_presigned_url = put_url_response["put_presigned_url"]
    s3_response = requests.put(
        s3_presigned_url,
        headers={"Content-Type": mime_type},
        data=fp,
        verify=False,
    )
    if s3_response.status_code != 200:
        logging.error("s3 upload failed: %s", s3_response.text)
        raise api.CollimatorApiError(
            f"Failed to upload file {name} to project {project_uuid}"
        )
    file_uuid = put_url_response["summary"]["uuid"]
    process_response = api.call(
        f"/projects/{project_uuid}/files/{file_uuid}/process", "POST"
    )
    return process_response["summary"]


def upload_files(project_uuid: str, files: list[str], overwrite=True):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for file in files:
            with open(file, "rb") as fp:
                futures.append(
                    executor.submit(
                        upload_file, project_uuid, os.path.basename(file), fp, overwrite
                    )
                )
        for future in concurrent.futures.as_completed(futures):
            future.result()
