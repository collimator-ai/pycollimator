import traceback

# import sys


def quiet_hook(kind, message, tb):
    traceback.print_tb(tb, limit=1)
    print(
        "{0}: {1}".format(kind.__name__, message)
    )  # Only print Error Type and Message


# sys.excepthook = quiet_hook


class Error(Exception):
    pass


class NoModelError(Error):
    """No model matches the string given"""

    def __init__(self, name):
        message = (
            "There are no models with the name "
            + name
            + ".  Please check your model editor and try again."
        )
        super().__init__(message)


class NoModelsError(Error):
    """No models are part of this project"""

    def __init__(self):
        message = "There are no models in this project.  Please add one to load in the notebook."
        super().__init__(message)


class NoModelParamsError(Error):
    def __init__(self):
        message = "There are no model parameters for this model.  Please add one through the model editor."
        super().__init__(message)


class ModelNotValidError(Error):
    def __init__(self):
        message = "This model is not valid for simulation.  Please correct the model through the model editor."
        super().__init__(message)


class NotFoundError(Error):
    def __init__(self, message: str = None):
        super().__init__(message or ("Not found"))


class NotLoadedError(Error):
    def __init__(self, message: str = None):
        super().__init__(message or ("Not loaded"))


class UnsupportedOperationError(Error):
    def __init__(self, message: str = None):
        super().__init__(message or ("Unsupported operation"))


class CollimatorApiError(Error):
    def __init__(self, message: str = None):
        super().__init__(message or ("Collimator API Error"))


class CollimatorRetryableApiError(Error):
    def __init__(self, message: str = None):
        super().__init__(message or ("Collimator API Error"))


class CollimatorRuntimeError(Error):
    def __init__(self, simulation_status_json):
        message = (
            simulation_status_json["fail_reason"]
            + "Please return to model editor to correct."
        )
        super().__init__(message)


class CollimatorAuthenticationError(Error):
    def __init__(self, message):
        message = (
            message
            or "Jupyter authentication has expired.  Please reload jupyter to reauthenticate user."
        )
        super().__init__(message)


class CollimatorUnexpectedError(Error):
    def __init__(self, msg: str = None):
        message = ("An unexpected error occured") + (msg is None and "." or ": ") + msg
        super().__init__(message)
