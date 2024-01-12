import json
import math


import numpy as np
import pandas as pd
import typing as T

from simpleeval import EvalWithCompoundTypes

from pycollimator.api import Api
from pycollimator.diagrams import ModelGraph, BlockPath
from pycollimator.error import NotFoundError, NotLoadedError
from pycollimator.global_variables import GlobalVariables
from pycollimator.log import Log
from pycollimator.projects import get_project
from pycollimator.simulation_hashed_file import SimulationHashedFile
from pycollimator.utils import is_path, is_uuid

# from collimator.simulations import run_simulation


# TODO pull these defaults from the schemas
# TODO add support for all model configuration parameters
DEFAULT_STOP_TIME = 10.0
DEFAULT_SAMPLE_TIME = 0.1
DEFAULT_INTERPOLATION = 0.01


class ModelConfiguration:
    def __init__(self, data: T.Any, model: "Model"):
        self.model = model
        self._data = data or {}
        self._stop_time = float(self._data.get("stop_time", DEFAULT_STOP_TIME))
        self._discrete_step = float(self._data.get("sample_time", DEFAULT_SAMPLE_TIME))
        self._interpolation = float(
            self._data.get("continuous_time_result_interval", DEFAULT_INTERPOLATION)
        )
        self._needs_resync = False
        # TODO parse events and other solver settings

    # FIXME use a proper serializer
    def to_dict(self) -> dict:
        return {
            **self._data,
            "stop_time": self.stop_time,
            "sample_time": self.discrete_step,
            "continuous_time_result_interval": self.interpolation,
        }

    def __repr__(self) -> str:
        d = self.to_dict()
        if d.get("__developer_options", None) is not None and not Log.is_level_above(
            "TRACE"
        ):
            del d["__developer_options"]
        return d.__repr__()

    def __getitem__(self, key):
        if key == "stop_time":
            return self.stop_time
        elif key == "discrete_step":
            return self.discrete_step
        elif key == "interpolation":
            return self.interpolation
        else:
            raise KeyError(
                "key must be one of 'stop_time', 'discrete_step', 'interpolation'"
            )

    def __setitem__(self, key, value: T.Any):
        if key == "stop_time":
            self.stop_time = value
        elif key == "discrete_step":
            self.discrete_step = value
        elif key == "interpolation":
            self.interpolation = value
        else:
            raise KeyError(
                "key must be one of 'stop_time', 'discrete_step', 'interpolation'"
            )

    @property
    def stop_time(self) -> float:
        return self._stop_time

    @stop_time.setter
    def stop_time(self, value: float):
        if self._stop_time != value and value > 0:
            self._stop_time = value
            self._needs_resync = True

    @property
    def discrete_step(self) -> float:
        return self._discrete_step

    @discrete_step.setter
    def discrete_step(self, value: float):
        if self._discrete_step != value and value > 0:
            self._discrete_step = value
            self._needs_resync = True

    @property
    def interpolation(self) -> float:
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: float):
        if self._interpolation != value and value > 0:
            self._interpolation = value
            self._needs_resync = True

    # def sync(self, remote_model_data: dict = None):
    #     if not self._needs_resync:
    #         return
    #     # should we merge with the remote model data?
    #     request_data = {
    #         "configuration": self.to_dict(),
    #     }
    #     resp = Api.model_configuration_update(self.model.uuid, request_data)
    #     self._needs_resync = False
    #     return resp


class ParameterEvaluator:
    @classmethod
    def __for_math_names(cls) -> T.Dict[str, T.Any]:
        for name in dir(math):
            if not name.startswith("_") and not callable(getattr(math, name)):
                yield name, getattr(math, name)

    @classmethod
    def __for_math_functions(cls) -> T.Dict[str, T.Any]:
        for name in dir(math):
            if not name.startswith("_") and callable(getattr(math, name)):
                yield name, getattr(math, name)

    @classmethod
    def __math_names(cls) -> T.Dict[str, T.Any]:
        return {name: value for name, value in cls.__for_math_names()}

    @classmethod
    def __math_functions(cls) -> T.Dict[str, T.Any]:
        return {name: value for name, value in cls.__for_math_functions()}

    @classmethod
    def all_functions(cls) -> T.Dict[str, T.Any]:
        return {**cls.__math_functions()}

    @classmethod
    def all_names(cls) -> T.Dict[str, T.Any]:
        return {**cls.__math_names()}

    @classmethod
    def evaluate(cls, name: str, expression: str) -> T.Any:
        evaluator = EvalWithCompoundTypes(
            functions=cls.all_functions(), names=cls.all_names()
        )
        try:
            literal = evaluator.eval(expression)
            return literal
        except Exception as e:
            raise ValueError(
                (
                    f"Could not evaluate parameter '{name}' with expression '{expression}': {e}"
                )
            )


# A quick and simple way to print out numpy arrays in a way that cmlc can handle.
# This will not scale much but should get us somewhere.
def _serialize_numpy_array(param: np.ndarray) -> str:
    lst = param.tolist()
    s = json.dumps(
        lst,
        ensure_ascii=True,
        allow_nan=True,
        indent=None,
        separators=(",", ":"),
    )
    # return f"np.array({s})"
    return s


class Parameter:
    def __init__(self, name, expression, is_string=False):
        self.name = name
        self.expression = expression
        self.is_string = is_string

    @property
    def value(self) -> T.Any:
        if self.is_string:
            return self.expression
        return ParameterEvaluator.evaluate(self.name, self.expression)

    def __str__(self) -> str:
        if self.is_string:
            return str(self.expression)
        if isinstance(self.expression, np.ndarray):
            return _serialize_numpy_array(self.expression)
        return str(self.expression)

    def __repr__(self) -> str:
        return f"'{self.expression}'"

    def to_api_data(self) -> T.Dict:
        if isinstance(self.expression, np.ndarray):
            return {"value": _serialize_numpy_array(self.expression)}
        elif self.is_string:
            return {"value": self.expression, "is_string": True}
        else:
            return {"value": str(self.expression)}


class ParameterSet:
    def __init__(self, data: dict = None):
        self._data = data or {}
        self._parameters: T.Dict[str, Parameter] = {}
        for name, param in data.items():  # type: str, dict
            if isinstance(param, dict):
                self._parameters[name] = Parameter(
                    name,
                    param.get("value", None),
                    param.get("is_string", False),
                )
            elif isinstance(param, str):
                self._parameters[name] = Parameter(name, param, is_string=True)
            else:
                self._parameters[name] = Parameter(name, param)

    def __repr__(self) -> str:
        return self._parameters.__repr__()

    def __getitem__(self, key):
        try:
            return self._parameters[key]
        except KeyError:
            raise NotFoundError(f"Model parameter '{key}' not found")

    def __setitem__(self, key, value: T.Any):
        return self.set(key, value)

    def keys(self):
        return self._parameters.keys()

    def items(self):
        return self._parameters.items()

    def values(self):
        return self._parameters.values()

    # def __setattr__(self, key: str, value: T.Any) -> None:
    #     if key.startswith("_"):
    #         super().__setattr__(key, value)
    #     return self.set(key, value)

    def to_api_data(self) -> dict:
        return {name: param.to_api_data() for name, param in self._parameters.items()}

    def get_value(self, name: str) -> T.Any:
        return self._parameters[name].value

    def set(self, name: str, literal: T.Any):
        try:
            self._parameters[name].expression = str(literal)
        except KeyError:
            raise NotFoundError(f"Model parameter '{name}' not found")


class ModelParameters(ParameterSet):
    def __init__(self, data: dict):
        super().__init__(data)


class Model:
    """
    Contents of a model, may be partially loaded.
    """

    def __init__(self, data):
        # TODO: for modifications, we'll need full representation of model.
        # We also already fully load models whenever we access a block. Might as well do it here.
        self._data = data
        self._graph: ModelGraph = None
        self._datasource_data_by_name_path: T.Dict[
            str, pd.DataFrame
        ] = {}  # submodel.block.path -> data (pd.DataFrame)
        self._hashed_files_by_name_path: T.Dict[
            str, SimulationHashedFile
        ] = {}  # submodel.block.path -> SimulationHashedFile
        self.path_data_by_named_path: T.Dict[
            str, BlockPath
        ] = {}  # submodel.block.path -> BlockPath
        self._configuration = ModelConfiguration(data.get("configuration", {}), self)
        self._parameters = ModelParameters(data.get("parameters", {}))

    def __getitem__(self, key):
        return self._data[key]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if Log.is_level_above("DEBUG"):
            return (
                f"<{self.__class__.__name__} name='{self.name}' uuid='{self.uuid}' "
                + f"version={self.version} is_loaded={self.is_loaded}>"
            )
        return f"<{self.__class__.__name__} name='{self.name}'>"

    @property
    def name(self) -> str:
        return self._data["name"]

    @property
    def uuid(self) -> str:
        return self._data["uuid"]

    @property
    def project_uuid(self) -> str:
        return self._data["project_uuid"]

    @property
    def url(self) -> str:
        return (
            f"{GlobalVariables.url()}/projects/{self.project_uuid}/models/{self.uuid}"
        )

    @property
    def version(self):
        return self._data.get("version")

    @property
    def configuration(self) -> ModelConfiguration:
        return self._configuration

    @property
    def parameters(self) -> ModelParameters:
        return self._parameters

    @property
    def is_loaded(self) -> bool:
        """
        Returns True if the model is fully loaded.
        """
        return self._graph is not None
        # return self._data.get("diagram") is not None

    def load(self):
        """
        Load the model, if it's not already loaded
        """
        if not self.is_loaded:
            self.reload()

    def reload(self):
        """
        Reload the model, if it's loaded or not
        """
        data = Api.get_model(self.uuid)
        self._data = data
        self._graph = ModelGraph(data, model=self)
        self._configuration = ModelConfiguration(data.get("configuration"), self)
        self._parameters = ModelParameters(data.get("parameters", {}))

    def sync(self):
        """
        Synchronize local changes with the database and back
        """
        data = Api.get_model(self.uuid)
        self._data["version"] = data["version"]
        self._graph = ModelGraph(data, model=self)
        # resp = self.configuration.sync(data)
        # if resp is not None and resp.get("version") is not None:
        #     self._data["version"] = resp["version"]

    # todo traverse given path
    def _find_block_by_path(self, path: str, autoload=False):
        """
        Find a block by path name
        """
        if not self.is_loaded:
            if not autoload:
                raise NotLoadedError((f"Model '{self}' is not loaded"))
            self.load()
        return self._graph.find_block_by_path(path)

    # todo: do a search the model graph that constructs path 2
    # todo: update so that one word argument is treated as a path.
    # e.g. 'Adder_0' is a name by default. Must say find_block(path='Adder_0') to force path
    def find_block(
        self,
        pattern: str = None,
        name: str = None,
        path: str = None,
        type: str = None,
        case=False,
        autoload=True,
    ):
        """
        Find a block by name or path.
        """

        if path:
            return self._find_block_by_path(path, autoload=autoload)
        elif is_path(pattern):
            return self._find_block_by_path(pattern, autoload=autoload)

        # possibly multiple
        blocks = self.find_blocks(
            pattern=pattern, name=name, type=type, case=case, autoload=autoload
        )
        if len(blocks) == 0:
            raise NotFoundError(
                (
                    f"No block was found in model '{self}' "
                    f"(pattern='{pattern}' name='{name}' path='{path}' type='{type}')"
                )
            )
        if len(blocks) > 1:
            raise NotFoundError(
                (
                    f"Multiple matching blocks found in model '{self}' "
                    f"(pattern='{pattern}' name='{name}' path='{path}' type='{type}'): {blocks}"
                )
            )
        return blocks[0]

    # todo: do a search the model graph that constructs path 3
    def find_blocks(
        self,
        pattern: str = None,
        name: str = None,
        type: str = None,
        case=False,
        autoload=True,
    ):
        """
        Find all blocks that match
        """
        if not self.is_loaded:
            if not autoload:
                raise NotLoadedError((f"Model '{self}' is not loaded"))
            self.load()
        return self._graph.find_blocks(pattern=pattern, name=name, type=type, case=case)

    def get_block_path(self, block, autoload=True):
        if not self.is_loaded:
            if not autoload:
                raise NotLoadedError((f"Model '{self}' is not loaded"))
            self.load()
        return self._graph.get_block_path(block)

    # Circular imports blah blah blah
    # def run(self, parameters: dict = None, wait=True, no_sync=False, ignore_cache=False):
    #     """
    #     Start a simulation.

    #     This is an alias for collimator.run_simulation.
    #     """
    #     return run_simulation(self, parameters=parameters, wait=wait, no_sync=no_sync, ignore_cache=ignore_cache)

    def set_parameters(self, parameters: dict, save=False):
        """Set model parameters.

        :param parameters: A dictionary of the model parameters and their desired values.
            Does not create new parameters.
        :type parameters: dict
        :param save: Option to save the parameter changes to the model, defaults to False
        :type save: bool, optional
        """
        paramset = ParameterSet(parameters)
        for k, v in paramset.items():
            self._parameters[k] = v
        if save:
            request_data = {"parameters": self.parameters.to_api_data()}
            Api.model_parameter_update(self.uuid, request_data)

    def get_parameters(self) -> ModelParameters:
        return self._parameters

    def set_configuration(self, configuration: dict = None, **kwargs):
        if configuration is not None:
            for k, v in configuration.items():
                self.configuration[k] = v
        for k, v in kwargs.items():
            self.configuration[k] = v

    def get_configuration(self) -> ModelConfiguration:
        return self._configuration

    def _set_datasource_data(self, block, data: pd.DataFrame):
        self.path_data_by_named_path[block.path] = block.path_data
        self._datasource_data_by_name_path[block.path] = data
        self._hashed_files_by_name_path.clear()

    def _get_blocks_with_data(self) -> T.Dict[str, pd.DataFrame]:
        """block_uuid -> data (pd.DataFrame)"""
        return self._datasource_data_by_name_path

    def _prepare_hashed_files(self) -> T.Dict[str, SimulationHashedFile]:
        for path in self._datasource_data_by_name_path:
            Log.trace(f"Preparing hashed files for block '{path}'")
            block_data = self._datasource_data_by_name_path[path]
            sdf = SimulationHashedFile(block_data)
            self._hashed_files_by_name_path[path] = sdf
        return self._hashed_files_by_name_path

    def _get_hashed_files(self) -> T.Dict[str, SimulationHashedFile]:
        if len(self._hashed_files_by_name_path) == 0:
            self._prepare_hashed_files()
        return self._hashed_files_by_name_path or {}


def load_model(name: str, project: str = None, load=True, case=False) -> Model:
    """
    Get a model from the API by name
    """
    if isinstance(name, Model):
        if load:
            name.load()
        return name

    if is_uuid(name):
        Log.debug(f"Getting model by uuid '{name}'")
        return Model(Api.get_model(name))

    prj = get_project(project)
    for m in prj["models"]:  # type: dict
        Log.trace(f"Checking model '{json.dumps(m)}'")
        if m.get("name") == name:
            model = Model(m)
            if load:
                model.load()
            return model

    if not case:
        Log.debug(
            f"No model named '{name}' found in project '{project}', trying case insentitive search"
        )
        for m in prj["models"]:  # type: dict
            if m.get("name", "").lower() == name.lower():
                model = Model(m)
                if load:
                    model.load()
                return model

    raise NotFoundError((f"Model not found: '{name}'"))


def list_models() -> T.List[Model]:
    """
    list all models available to be loaded in to the project
    """
    model_list_json = Api.get_project()
    return [
        Model(model_list_json["models"][i])
        for i in range(len(model_list_json["models"]))
    ]


def write_model_json(m) -> Model:
    project_uuid = GlobalVariables.project_uuid()
    m["project_uuid"] = project_uuid
    for e in list_models():
        if m["name"] == e["name"]:
            uuid = e["uuid"]
            m["version"] = e["version"]
            print("replacing uuid", uuid, "version", m["version"])
            Api.put(f"/api/v0/models/{uuid}", m)
            return load_model(uuid)
    res = Api.post("/api/v0/models", m)
    uuid = res["uuid"]
    return load_model(uuid)
