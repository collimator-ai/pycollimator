class WildcatError(Exception):
    """Base class for all custom Wildcat errors."""

    def __init__(self, message=None, block_id: str = None):
        super().__init__(message)
        self.message = message
        self.block_id = block_id


class StaticError(WildcatError):
    """Wraps a Python exception to record the offending block id. The original
    exception is found in the '__cause__' field.

    See lynx.framework.context_factory._check_types for use.

    This is call 'static' (as opposed to say 'runtime') meaning this is for
    wrapping errors detected prior to running a simulation."""

    def __init__(self, block_id, message: str = None):
        super().__init__(message=message, block_id=block_id)


class ShapeMismatchError(StaticError):
    """Block parameters or input/outputs have mismatched shapes."""

    def __init__(self, block_id):
        super().__init__(block_id=block_id)


class DtypeMismatchError(StaticError):
    """Block parameters or input/outputs have mismatched dtypes."""

    def __init__(self, block_id):
        super().__init__(block_id=block_id)


class BlockInitializationError(WildcatError):
    """A generic error to be thrown when a block fails at init time, but
    the full exceptions are known to cause issues, eg. with ray serialization.
    """

    def __init__(self, block_id, message: str):
        super().__init__(message=message, block_id=block_id)


class BlockRuntimeError(WildcatError):
    """A generic error to be thrown when a block fails at runtime, but
    the full exceptions are known to cause issues, eg. with ray serialization.
    """

    def __init__(self, block_id, message: str):
        super().__init__(message=message, block_id=block_id)
