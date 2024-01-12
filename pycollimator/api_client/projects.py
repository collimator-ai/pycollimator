from .api import Api
from .error import NotFoundError
from .log import Log
from .utils import is_uuid


class Project:
    def __init__(self, data):
        self._data = data
        # api_dict_safe_copy(self.__dict__, data)

    def __getitem__(self, key):
        return self._data[key]

    def __str__(self) -> str:
        return self["title"]

    def __repr__(self) -> str:
        if Log.is_level_above("DEBUG"):
            return f"<{self.__class__.__name__} title='{self['title']}' uuid='{self['uuid']}'>"
        return f"<{self.__class__.__name__} title='{self['title']}'>"

    @property
    def uuid(self) -> str:
        return self["uuid"]


def get_project(title: str = None, case=False) -> Project:
    """
    Get the current project or another project by name.
    """

    prj = None
    if title is None:
        prj = Api.get_project()
    elif is_uuid(title):
        prj = Api.get_project(title)
    else:
        projects = Api.get_projects()
        for p in projects["projects"]:
            if p.get("title") == title:
                prj = p
                break
        if not case:
            for p in projects["projects"]:
                if p.get("title", "").lower() == title.lower():
                    prj = p
                    break

    if prj is None:
        raise NotFoundError((f"Project not found: '{title}'"))

    return Project(prj)
