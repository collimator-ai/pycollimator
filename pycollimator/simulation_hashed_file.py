import typing as T
import pandas as pd
import tempfile
import os


from pycollimator.api import Api
from pycollimator.error import CollimatorUnexpectedError
from pycollimator.hash import Hash
from pycollimator.log import Log


class SimulationHashedFile:
    def __init__(self, data: T.Any) -> None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Unsupported data type. Must be: pandas.DataFrame")
        self._hash = None
        self.file = None
        self.data = data
        self.data.index.name = "time"
        self.content_type = "text/csv"

    @property
    def hash(self) -> str:
        self._ensure_write()
        if self._hash is None:
            self._hash = Hash.sha256sum(self.file.name)
        return self._hash

    def upload(self, model_uuid: str, simulation_uuid: str) -> None:
        self._ensure_write()
        # Uses hashed file name to indicate EFS file to pull data from.
        return self._do_upload(model_uuid, simulation_uuid, self.hash, self.file.name)

    def _ensure_write(self):
        if self.file is not None:
            return
        self.file = tempfile.NamedTemporaryFile(
            prefix=".simdata-", suffix=".csv", mode="w+", dir=".", delete=False
        )
        # Close the file because Windows can't have two handles open at once.
        # FIXME: we should clean up the file after upload
        self.file.close()
        Log.debug(f"Writing csv data to {self.file.name}")
        self.data.to_csv(self.file.name, index=True, header=True)

    def _do_upload(
        self,
        model_uuid: str,
        simulation_uuid: str,
        hash_value: str,
        path: T.Any,
    ) -> None:
        Log.debug(f"Creating simulation hashed file {path} with hash {hash_value}")
        content_length = os.path.getsize(path)
        body = {
            "hash_type": "sha256",
            "hash_value": hash_value,
            "content_type": "text/csv",
            "content_length": content_length,
        }
        response = Api.simulation_hashed_file_create(model_uuid, simulation_uuid, body)
        if response.get("in_cache", False) is True:
            Log.debug("Simulation hashed file is already in cache")
        elif response.get("upload_url", None) is not None:
            Log.debug(
                "Uploading simulation hashed file to url:",
                response["upload_url"],
            )
            response = Api.simulation_hashed_file_upload(path, response["upload_url"])
        else:
            raise CollimatorUnexpectedError(
                "file is not cached but no upload url was returned"
            )
        return response
