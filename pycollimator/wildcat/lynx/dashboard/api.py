import dataclasses
import json
import logging
import os
import requests
from typing import Any
from uuid import uuid4


logger = logging.getLogger(__name__)

_API_BASE_URL = os.environ.get("COLLIMATOR_API_URL", "https://app.collimator.ai/api/v0")
_API_TOKEN = os.environ.get("COLLIMATOR_API_TOKEN", None)

_HEADERS = {
    "X-Collimator-API-Caller": "pycollimator",
    "X-Request-ID": str(uuid4()),
    "Accept": "application/json",
    "X-Collimator-API-Token": _API_TOKEN,
}


class CollimatorApiError(Exception):
    ...


class CollimatorRetryableApiError(CollimatorApiError):
    ...


class CollimatorAuthenticationError(CollimatorApiError):
    ...


class CollimatorNotFoundError(CollimatorApiError):
    ...


def _handle_error(response: requests.Response, headers: dict[str, str] = None):
    request_id = (headers is not None) and headers.get("X-Request-ID") or None
    if response.status_code >= 400:
        if response.status_code == 401:
            logger.debug("API error 401, response: %s", response.text)
            raise CollimatorAuthenticationError(
                "Authentication failed. Please check your token or reload the page."
            )
        elif response.status_code == 504:
            logger.debug("API error 504, retrying...")
            raise CollimatorRetryableApiError
        elif response.status_code == 404:
            raise CollimatorNotFoundError
        else:
            logger.debug(
                "API error, status_code: %s, response: %s",
                response.status_code,
                response.text,
            )
            raise CollimatorApiError(
                f"API call failed (status: {response.status_code}). Support request ID: "
                f"{request_id}. Detailed error:\n{response.text}"
            )


def _convert_dataclasses_to_dict(d: Any):
    if isinstance(d, dict):
        return {k: _convert_dataclasses_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_convert_dataclasses_to_dict(v) for v in d]
    elif dataclasses.is_dataclass(d):
        return _convert_dataclasses_to_dict(dataclasses.asdict(d))
    else:
        return d


def call(
    api,
    method,
    body: dict = None,
    headers: dict = None,
    body_type="json",
    response_type="json",
    retries=3,
):
    url = f"{_API_BASE_URL}{api}"

    if not headers:
        headers = {}

    headers = {**_HEADERS, **headers}
    body = _convert_dataclasses_to_dict(body)
    timeout = (5, 30)
    try:
        if body is None:
            logger.debug("%s %s", method, url)
            response = requests.request(method, url, headers=headers, timeout=timeout)
        elif body_type == "json":
            logger.debug("%s %s %s", method, url, json.dumps(body, indent=2))
            headers["Content-Type"] = "application/json"
            response = requests.request(
                method, url, json=body, headers=headers, timeout=timeout
            )
        elif body_type == "files":
            logger.debug("%s %s %s", method, url, body)
            response = requests.request(
                method, url, files=body, headers=headers, timeout=timeout
            )
        _handle_error(response, headers)
    except CollimatorRetryableApiError as e:
        if retries > 0:
            logger.debug("%s retrying...", method)
            return call(
                api,
                method,
                body,
                headers,
                body_type,
                response_type,
                retries - 1,
            )
        raise e

    logger.debug(
        "method: %s, response: %s, status: %s",
        method,
        response.text,
        response.status_code,
    )
    if response.status_code != requests.codes["no_content"]:
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
