import mimetypes
import os
import typing as T
import requests

from pycollimator.log import Log
from pycollimator.error import (
    CollimatorAuthenticationError,
    CollimatorApiError,
    CollimatorRetryableApiError,
)
from pycollimator.global_variables import GlobalVariables


class Api:
    BASE_PATH = "/api/v0"

    @classmethod
    def _handle_error(cls, response: T.Any, headers: T.Dict[str, str] = None):
        request_id = (headers is not None) and headers.get("X-Request-ID") or None
        # correlation_id = headers and headers.get('X-Correlation-ID') or None
        if response.status_code >= 400:
            if response.status_code == 401:
                Log.debug("API error 401, response:", response.text)
                raise CollimatorAuthenticationError(
                    (
                        "Authentication failed. Please check your token or reload the page."
                    )
                )
            elif response.status_code == 504:
                Log.trace("API error 504, retrying...")
                raise CollimatorRetryableApiError
            else:
                Log.debug(
                    "API error, status_code:",
                    response.status_code,
                    "response:",
                    response.text,
                )
                raise CollimatorApiError(
                    f"API call failed (status: {response.status_code}). Support request ID: "
                    + f"{request_id}. Detailed error:\n{response.text}"
                )

    # TODO add streaming support for logs and results
    @classmethod
    def _call_api(
        cls,
        api,
        method,
        body: T.Any = None,
        headers: dict = None,
        body_type="json",
        response_type="json",
        retries=3,
    ):
        try:
            url = GlobalVariables.url() + api
            headers = headers or GlobalVariables.custom_headers()
            timeout = (5, 30)

            if body is None:
                Log.trace(method, url)
                response = requests.request(
                    method, url, headers=headers, timeout=timeout
                )
            elif body_type == "json":
                Log.trace(method, url, body)
                headers["Content-Type"] = "application/json"
                response = requests.request(
                    method, url, json=body, headers=headers, timeout=timeout
                )
            elif body_type == "files":
                Log.trace(method, url, body)
                response = requests.request(
                    method, url, files=body, headers=headers, timeout=timeout
                )
            cls._handle_error(response, headers)

            Log.trace(method, "response:", response.text)
            Log.trace(method, "response:", response.status_code)
            if response.status_code not in (
                requests.codes["no_content"],
                requests.codes["created"],
            ):
                if response_type == "text":
                    return response.text
                elif response.headers.get("content-type") and response.headers[
                    "content-type"
                ].strip().startswith("application/json"):
                    return response.json()
                else:
                    raise CollimatorApiError(
                        f"Unexpected response type (status: {response.status_code}). Support request ID: "
                        + f"{headers['X-Request-ID'] }. Detailed error:\n{response.text}"
                    )

        except CollimatorRetryableApiError as e:
            if retries > 0:
                Log.trace(method, "retrying...")
                return cls._call_api(
                    api,
                    method,
                    body,
                    headers,
                    body_type,
                    response_type,
                    retries - 1,
                )
            raise e

    @classmethod
    def get(cls, api):
        return cls._call_api(api, method="GET")

    @classmethod
    def post(cls, api, body):
        return cls._call_api(api, method="POST", body=body)

    @classmethod
    def put(cls, api, body):
        return cls._call_api(api, method="PUT", body=body)

    @classmethod
    def delete(cls, api, body):
        return cls._call_api(api, method="DELETE", body=body)

    @classmethod
    def get_projects(cls):
        api = f"{Api.BASE_PATH}/projects"
        return cls.get(api)

    @classmethod
    def get_project(cls, project_uuid: str = None) -> dict:
        project_uuid = project_uuid or GlobalVariables.project_uuid()
        api = f"{Api.BASE_PATH}/projects/{project_uuid}"
        return cls.get(api)

    @classmethod
    def get_project_by_name(cls, name: str) -> dict:
        projects = cls.get_projects()["projects"]
        for p in projects:
            if p["title"] == name:
                return p
        return {}

    @classmethod
    def get_model(cls, model_uuid: str) -> dict:
        api = f"{Api.BASE_PATH}/models/{model_uuid}"
        return cls.get(api)

    @classmethod
    def get_submodel(cls, submodel_uuid: str, *, project_uuid: str = None) -> dict:
        project_uuid = project_uuid or GlobalVariables.project_uuid()
        api = f"{Api.BASE_PATH}/project/{project_uuid}/submodels/{submodel_uuid}"
        return cls.get(api)

    @classmethod
    def put_model(cls, model: dict):
        api = f"{Api.BASE_PATH}/models/{model['uuid']}"
        return cls.put(api, model)

    @classmethod
    def post_model(cls, model: dict):
        api = f"{Api.BASE_PATH}/models"
        return cls.post(api, model)

    @classmethod
    def simulation_create(cls, model_uuid: str, body: dict) -> dict:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations"
        return cls.post(api, body)

    @classmethod
    def simulation_get(cls, model_uuid: str, simulation_uuid: str) -> dict:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}"
        return cls.get(api)

    @classmethod
    def simulation_start(cls, model_uuid: str, simulation_uuid: str) -> dict:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}/start"
        return cls.post(api, None)

    @classmethod
    def simulation_stop(cls, model_uuid: str, simulation_uuid: str) -> dict:
        api = (
            f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}/events"
        )
        return cls.post(api, {"command": "stop"})

    @classmethod
    def simulation_parameters_set(
        cls, model_uuid: str, simulation_uuid: str, body: dict
    ) -> dict:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}/parameters"
        return cls.put(api, body)

    @classmethod
    def simulation_update(
        cls, model_uuid: str, simulation_uuid: str, body: dict
    ) -> dict:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}"
        return cls.put(api, body)

    @classmethod
    def simulation_hashed_file_create(
        cls, model_uuid: str, simulation_uuid: str, body: dict
    ) -> dict:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}/hashed_files"
        return cls.post(api, body)

    @classmethod
    def simulation_hashed_file_upload(cls, path: str, upload_url: str) -> dict:
        with open(path, "rb") as file:
            return cls._call_api(
                upload_url, method="PUT", body={"file": file}, body_type="files"
            )

    # FIXME should be a stream
    @classmethod
    def simulation_logs(cls, model_uuid: str, simulation_uuid: str) -> str:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}/logs"
        return cls._call_api(api, method="GET", response_type="text")

    # FIXME should be a stream
    @classmethod
    def simulation_results(
        cls, model_uuid: str, simulation_uuid: str, retries=3
    ) -> str:
        name = "continuous_results.csv"
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}"
        api += f"/process_results?files={name}"

        s3_urls_batch_resp = cls._call_api(api, method="GET", retries=retries)
        urls = cls._get_result_urls(s3_urls_batch_resp, [name])

        return cls._get_file(urls[0])

    @classmethod
    def signal_results(
        cls,
        model_uuid: str,
        simulation_uuid: str,
        paths: T.List[str],
        retries=3,
    ) -> T.List[str]:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}"
        api += f"/process_results?files={','.join(paths)}"

        s3_urls_batch_resp = cls._call_api(api, method="GET", retries=retries)
        urls = cls._get_result_urls(s3_urls_batch_resp, paths)

        # TODO: async fetch if users start specifying many signals
        return [cls._get_file(url) for url in urls]

    # @classmethod
    # def model_configuration_update(cls, model_uuid: str, body: dict) -> dict:
    #     api = f"{Api.BASE_PATH}/models/{model_uuid}/configuration"
    #     return cls.put(api, body)

    @classmethod
    def model_parameter_update(cls, model_uuid: str, body: dict) -> dict:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/parameters"
        return cls.put(api, body)

    @classmethod
    def linearization_results_csv(cls, model_uuid: str, simulation_uuid: str) -> str:
        api = f"{Api.BASE_PATH}/models/{model_uuid}/simulations/{simulation_uuid}/lin_results"
        return cls._call_api(api, method="GET", response_type="text")

    @classmethod
    def upload_file(
        cls, filepath: str, project_uuid: str = None, overwrite: bool = False
    ) -> str:
        project_uuid = project_uuid or GlobalVariables.project_uuid()
        content_type, _ = mimetypes.guess_type(filepath)
        size = os.stat(filepath).st_size

        response = cls._call_api(
            f"{Api.BASE_PATH}/projects/{project_uuid}/files",
            method="POST",
            body={
                "name": os.path.basename(filepath),
                "content_type": content_type or "application/octet-stream",
                "size": size,
                "overwrite": overwrite,
            },
        )

        file_uuid = response["summary"]["uuid"]
        s3_url = response["put_presigned_url"]

        with open(filepath, "rb") as f:
            response = requests.put(
                s3_url,
                data=f,
                headers={
                    "Content-Type": content_type,
                    "Content-Length": str(size),
                },
            )

        if response.status_code != 200:
            Log.debug(
                "API error, status_code:",
                response.status_code,
                "response:",
                response.json,
            )
            raise CollimatorApiError(
                f"Failed to upload to S3. Detailed error:\n{response.text}"
            )

        return cls._call_api(
            f"{Api.BASE_PATH}/projects/{project_uuid}/files/{file_uuid}/process",
            method="POST",
        )

    @staticmethod
    def _get_result_urls(batch_resp: dict, names: T.List[str]) -> T.List[str]:
        errors = []
        urls = []

        for s3_url_resp in batch_resp["s3_urls"]:
            name = s3_url_resp.get("name")
            if name not in names:
                continue
            url = s3_url_resp.get("url")
            if url is None:
                errors.append(
                    f"Error for signal '{s3_url_resp.get('name')}'\n{s3_url_resp.get('error')}"
                )
            else:
                urls.append(url)

        if len(errors) > 0:
            raise CollimatorApiError("\n".join(errors))

        return urls

    @classmethod
    def _get_file(cls, url: str) -> str:
        response = requests.request("GET", url)
        cls._handle_error(response)
        return response.text
