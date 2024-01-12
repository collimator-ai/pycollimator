import dataclasses
import enum


# TODO: these dataclasses should be generated from openapi.yaml spec
class ModelKind(enum.StrEnum):
    MODEL = "Model"
    SUBMODEL = "Submodel"


@dataclasses.dataclass
class ModelSummary:
    uuid: str
    kind: ModelKind
    name: str


@dataclasses.dataclass
class FileSummary:
    uuid: str
    name: str  # url
    status: str


@dataclasses.dataclass
class ProjectSummary:
    uuid: str
    title: str
    models: list[ModelSummary]
    reference_submodels: list[ModelSummary]
    files: list[FileSummary]
