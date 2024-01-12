import typing as T
from uuid import uuid4

from pycollimator.log import Log
from pycollimator.utils import is_pyodide


class GlobalVariables:
    """
    global variables stores information about user's authentication and the project
    folder they are currently in
    """

    _instance: "GlobalVariables" = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(GlobalVariables, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.api_url = "https://app.collimator.ai"
        self.auth_token = None
        self.project = None
        self.target = "pycollimator"
        self.model = None
        self.version_id = None

    def set_auth_token(
        self, token: str, project: str = None, api_url: str = None
    ) -> None:
        if api_url is None:
            api_url = "https://app.collimator.ai"
        self.auth_token = token
        self.project = project
        self.api_url = api_url.rstrip("/")
        self.target = "pycollimator"
        Log.trace(
            "auth_token:",
            self.auth_token,
            "project:",
            self.project,
            "api_url:",
            self.api_url,
        )

    @classmethod
    def _get_instance(cls) -> "GlobalVariables":
        if cls._instance is None:
            cls()
        return cls._instance

    @classmethod
    def project_uuid(cls):
        """
        stores the project uuid associated with the folder.
        """
        return cls._get_instance().project

    @classmethod
    def model_uuid(cls):
        """
        stores the current model uuid.
        """
        return cls._get_instance().model

    @classmethod
    def model_version_id(cls):
        """
        stores the current model version id.
        """
        return cls._get_instance().version_id

    @classmethod
    def token(cls):
        """
        stores the authentication token
        """
        return cls._get_instance().auth_token

    @classmethod
    def url(cls):
        """
        stores the url, used for environment logic
        """
        return cls._get_instance().api_url

    @classmethod
    def custom_headers(cls) -> T.Dict[str, str]:
        """
        stores the authentication headers used in all API requests
        """

        headers = {
            "X-Collimator-API-Caller": "pycollimator",
            "X-Request-ID": str(uuid4()),
            "Accept": "application/json",
        }

        if is_pyodide():
            # No need for an auth token from the browser
            return headers

        return {"X-Collimator-API-Token": cls._get_instance().auth_token, **headers}


def get_project_url() -> str:
    return GlobalVariables.url() + "/projects/" + GlobalVariables.project_uuid()


def set_auth_token(token: str, project_uuid: str = None, api_url: str = None) -> None:
    GlobalVariables._get_instance().set_auth_token(token, project_uuid, api_url)
