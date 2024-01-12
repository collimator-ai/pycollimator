"""Utility functions for pycollimator"""

import uuid
import sys


PATH_DELIMITER = "."


def is_uuid(s: str) -> bool:
    """Returns True if the string is a uuid, False otherwise"""
    try:
        uuid.UUID(str(s), version=4)
        return True
    except ValueError:
        return False


# technically a nested path.
def is_path(s: str) -> bool:
    """Returns True if the string is a path, False otherwise"""
    return s and PATH_DELIMITER in s


def is_pyodide() -> bool:
    """Returns True if running in emscripten, False otherwise"""
    return sys.platform == "emscripten"
