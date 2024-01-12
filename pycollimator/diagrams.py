import re

import pandas as pd
from typing import List, Optional

from pycollimator.error import NotFoundError
from pycollimator.log import Log


# Current structure of SimulationModel:
# "diagram" : { ModelDiagram }
# "submodels": {
#   "references": {
#      "<uuid>": { "diagram_uuid": "<uuid>" },
#       ...
#   },
#   "diagrams": {
#     "<uuid>": { ModelDiagram }
#   }
# }


# FIXME FIXME FIXME
# We don't have a strong representation of the model, and we create
# Block objects on the fly. Modifying them means we need to backtrack
# to the model and update it somehow. Right now this scenario is limited
# to DataSourceBlock input data. We don't really want to allow modifying
# the model, so it's kinda okay. At least for now.


class BlockPath:
    def __init__(self, path: list, uuid_path: list):
        self.name_path = path
        self.uuid_path = uuid_path

    def get_name_path(self):
        return ".".join(self.name_path)

    def get_child_path(self, block_data) -> "BlockPath":
        return BlockPath(
            self.name_path + [block_data["name"]],
            self.uuid_path + [block_data["uuid"]],
        )


class Block:
    """
    Representation of a block in a model.

    Can be returned to the API user.
    """

    @classmethod
    def from_data(cls, data, model, path: BlockPath):
        if data.get("type") == "core.DataSource":
            return DataSourceBlock(data, model, path)
        return cls(data, model, path)

    def __init__(self, block_json, model, path):
        self.model = model
        self._json = block_json
        self._path = path

    def __getitem__(self, key):
        return self._json[key]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if Log.is_level_above("DEBUG"):
            return f"<{self.__class__.__name__} name='{self.name}' type='{self.type}' uuid='{self.uuid}'>"
        return f"<{self.__class__.__name__} name='{self.name}' type='{self.type}'>"

    @property
    def name(self):
        return self._json["name"]

    @property
    def uuid(self):
        return self._json["uuid"]

    @property
    def type(self):
        return self._json["type"]

    @property
    def path(self):
        return self._path.get_name_path()

    @property
    def path_data(self):
        return self._path

    # @property
    # def parameters(self):
    #     return self._json["parameters"]

    # TODO: expose setting block params

    def get_parameter(self, name: str, no_eval=False):
        param = self._json["parameters"].get(name)
        Log.trace(f"get_parameter: {name}={param}")
        if param is None:
            raise NotFoundError(
                (
                    f"Block '{self}' of type '{self.type}' does not have parameter '{name}'"
                )
            )
        if param.get("is_string", False) is True:
            return str(param["value"])
        if no_eval is True:
            return param["value"]
        expr = param.get("expression") or param["value"]
        evaluated = eval(expr)
        return evaluated


# Special block that reads csv
class DataSourceBlock(Block):
    def __init__(self, block_json, model, path: BlockPath):
        if block_json.get("type") != "core.DataSource":
            raise TypeError(("DataSourceBlock must be created from a DataSource block"))
        super().__init__(block_json, model, path)
        self.data = None

    def set_data(self, data: pd.DataFrame):
        # FIXME make sure the shape is correct and all that
        if not isinstance(data, pd.DataFrame):
            raise TypeError(("Input data must be a pandas DataFrame"))
        # set data of a DataSource block
        Log.trace("set_data, shape:", data.shape, "block:", self.__repr__())
        self.data = data.copy()
        self.data.index.name = "time"
        self.model._set_datasource_data(self, self.data)


class ModelDiagram:
    """
    Contents of a fully loaded model diagram (single plane).

    For use by internal APIs.
    """

    def __init__(self, data, model):
        self.model = model
        self.diagram = data

    def __str__(self) -> str:
        if self.diagram.get("name") is not None:
            return self.diagram["name"]
        return self.diagram["uuid"]

    def __repr__(self) -> str:
        if Log.is_level_above("DEBUG"):
            uid = self.diagram["uuid"]
            return f"<{self.__class__.__name__} model='{self.model}' uuid='{uid}'>"
        return f"<{self.__class__.__name__} model='{self.model}'>"

    @property
    def nodes(self):
        return self.diagram.get("nodes", [])

    @property
    def links(self):
        return self.diagram.get("links", [])

    # Full path only works at the model graph level.
    def find_block(
        self,
        pattern: str = None,
        name: str = None,
        type: str = None,
        ignorecase=True,
    ) -> Optional[Block]:
        blocks = self.find_blocks(
            pattern=pattern, name=name, type=type, ignorecase=ignorecase
        )
        if len(blocks) == 0:
            return None
        if len(blocks) > 1:
            raise NotFoundError(
                (f"Multiple blocks found for '{name}' in model '{self}'")
            )
        return blocks[0]

    # todo: do a search on model graph that constructs path 5
    # Not to be called with a path name as path name is unique.
    # probably won't need this ModelDiagram helper once BFS / path construction implemented at ModelGraph level
    def find_blocks(
        self,
        pattern: str = None,
        name: str = None,
        type: str = None,
        ignorecase=True,
        parent_path: BlockPath = None,
    ) -> List[Block]:
        blocks_found = []
        if parent_path is None:
            parent_path = BlockPath([], [])

        if name is None and pattern is None and type is None:
            pattern = ""  # matches any string

        if pattern is not None:
            rgx = re.compile(pattern, re.IGNORECASE if ignorecase else 0)
            for node in self.nodes:
                if rgx.match(node.get("name")):
                    path = parent_path.get_child_path(node)
                    blk = Block.from_data(node, self.model, path)
                    blocks_found.append(blk)

        if name is not None:
            for node in self.nodes:
                if (ignorecase and node.get("name", "").lower() == name.lower()) or (
                    node.get("name") == name
                ):
                    path = parent_path.get_child_path(node)
                    blk = Block.from_data(node, self.model, path)
                    blocks_found.append(blk)

        if type is None:
            return blocks_found

        if not type.startswith("core."):
            type = "core." + type
        type = type.lower()

        # If type is set: filter by type, or return all blocks of the given type if no other search criteria was set
        if name is not None or pattern is not None:
            return [blk for blk in blocks_found if blk.type.lower() == type]

        for node in self.nodes:
            if node.get("type").lower() == type:
                path = parent_path.get_child_path(node)
                blk = Block.from_data(node, self.model, path)
                blocks_found.append(blk)

        return blocks_found


class ModelGraph:
    """
    Contents of a fully loaded model graph (all loadable planes).

    For use by internal APIs.
    """

    def __init__(self, data, model):
        self._data = data
        self._model = model
        self.uuid = data.get("uuid")
        self.name = data.get("name")
        self.root_diagram = ModelDiagram(data["diagram"], model=self._model)
        self.diagrams_by_submodel_uuid = {}  # type: dict[str, ModelDiagram]

        submodel_diagrams = data.get("submodels", {}).get("diagrams", {})
        submodel_references = data.get("submodels", {}).get("references", {})
        self.submodel_references = submodel_references

        for submodel_uuid in self.submodel_references:
            ref = self.submodel_references[submodel_uuid]
            diagram_uuid = ref["diagram_uuid"]
            diagram = submodel_diagrams[diagram_uuid]
            self.diagrams_by_submodel_uuid[submodel_uuid] = ModelDiagram(diagram, model)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        if Log.is_level_above("DEBUG"):
            return (
                f"<{self.__class__.__name__} model='{self._model}' uuid='{self.uuid}'>"
            )
        return f"<{self.__class__.__name__} model='{self._model}'>"

    # todo add case flag.
    # todo generalize
    def find_block_by_path(self, path: str):
        diagram = self.root_diagram
        path_to_traverse = path.split(".")
        uuid_path = []

        for name in path_to_traverse[:-1]:
            matched_nodes = [node for node in diagram.nodes if node["name"] == name]
            if len(matched_nodes) == 0:
                raise NotFoundError((f"No block was found in path '{path}' "))
            if len(matched_nodes) > 1:
                raise NotFoundError(
                    (f"Multiple matching blocks found in path '{path}' ")
                )
            node = matched_nodes[0]
            if not (node["type"] == "core.Submodel" or node["type"] == "core.Group"):
                raise NotFoundError((f"No block was found in path '{path}' "))
            # move onto next diagram
            uuid_path.append(node["uuid"])
            diagram = self.diagrams_by_submodel_uuid[node["uuid"]]

        # last name in path can be any type of block
        last_name = path_to_traverse[-1]
        found_blocks = [node for node in diagram.nodes if node["name"] == last_name]

        if len(found_blocks) == 0:
            raise NotFoundError((f"No block was found in path '{path}' "))
        if len(found_blocks) > 1:
            raise NotFoundError((f"Multiple matching blocks found in path '{path}' "))

        uuid_path.append(found_blocks[0]["uuid"])
        block_path = BlockPath(path=path.split("."), uuid_path=uuid_path)
        return Block.from_data(found_blocks[0], diagram.model, block_path)

    # FIXME walk and construct blocks paths
    # For non-path search.
    # The ModelGraph is the 'root' class in which you can traverse. ModelDiagram is a node.
    # No cycles in graph/diagrams representation of model.
    # With submodels v2, UUIDs are not unique, so we must construct full paths for any block that is initialized.
    # Later, consider doing this upfront and only constructing full graph once.
    # todo: do a search the model graph that constructs path 4.
    def find_blocks(
        self, pattern: str = None, name: str = None, type: str = None, case=True
    ) -> List[Block]:
        found = []
        parent_path = BlockPath(path=[], uuid_path=[])
        diagrams_to_walk = [(self.root_diagram, parent_path)]
        safety_counter = 0

        while len(diagrams_to_walk) > 0 and safety_counter < 1000:
            safety_counter += 1
            entry = diagrams_to_walk.pop()
            diagram = entry[0]
            parent_path = entry[1]
            for node in diagram.nodes:
                if node["type"] == "core.Submodel" or node["type"] == "core.Group":
                    submodel_uuid = node["uuid"]
                    submodel_diagram = self.diagrams_by_submodel_uuid[submodel_uuid]
                    sub_path = parent_path.get_child_path(node)
                    sub_entry = (submodel_diagram, sub_path)
                    diagrams_to_walk.append(sub_entry)

                if pattern is not None:
                    if re.search(pattern, node["name"], re.IGNORECASE):
                        path = parent_path.get_child_path(node)
                        found.append(Block.from_data(node, diagram.model, path))

                if name is not None:
                    if node["name"] == name or (
                        not case and node["name"].lower() == name.lower()
                    ):
                        path = parent_path.get_child_path(node)
                        found.append(Block.from_data(node, diagram.model, path))

                if type is not None:
                    if not type.startswith("core."):
                        type = f"core.{type}"
                    if node["type"].lower() == type.lower():
                        path = parent_path.get_child_path(node)
                        found.append(Block.from_data(node, diagram.model, path))

        if safety_counter >= 1000:
            raise RuntimeError("Infinite loop detected while traversing model graph")

        return found

    def get_block_path(self, block):
        """
        Get the path of a block in the model.
        """
        if not isinstance(block, Block):
            blocks = self.find_blocks(name=block)
            if len(blocks) == 0:
                blocks = self.find_blocks(pattern=block)
            if len(blocks) == 0:
                raise NotFoundError((f"No block found for '{block}'"))
            if len(blocks) > 1:
                raise NotFoundError((f"Multiple blocks found for '{block}'"))
            return blocks[0].path
        return block.path
