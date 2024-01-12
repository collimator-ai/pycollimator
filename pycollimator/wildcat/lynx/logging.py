# https://www.firedrakeproject.org/_modules/firedrake/logging.html
import logging

from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

__all__ = [
    "logger",
    "set_log_level",
    "set_log_handlers",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

packages = [__package__, "wildcat_profiler"]


def set_log_handlers(handlers=None, to_file=None, pkg: str = None):
    """Set the log handlers for a specified or all packages."""

    if handlers is None:
        handlers = {}

    packages_ = packages if pkg is None else [pkg]
    for package in packages_:
        logger_ = logging.getLogger(package)
        for handler in logger.handlers:
            logger_.removeHandler(handler)

        handler = handlers.get(package, None)
        if handler is None:
            fmt = "%(name)s:%(levelname)s %(message)s"
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter(fmt=fmt))

            logger_.addHandler(sh)

            if to_file is not None:
                fh = logging.FileHandler(to_file, mode="w")
                fh.setFormatter(logging.Formatter(fmt=fmt))
                logger_.addHandler(fh)


def set_log_level(level, pkg: str = None):
    """Set the log level for the specified or all packages.

    Args:
        level: The log level to set.
        pkg: If set, apply the log level only to the specified package.
    """
    if pkg is not None:
        logger_ = logging.getLogger(pkg)
        logger_.setLevel(level)
        return

    for package in packages:
        logger_ = logging.getLogger(package)
        logger_.setLevel(level)


def scope_logging(func):
    """Decorator to log function entry and exit."""

    def wrapper(*args, **kwargs):
        logger_ = logging.getLogger(__package__)
        logger_.debug("*** Entering %s ***", func.__qualname__)
        result = func(*args, **kwargs)
        logger_.debug("*** Exiting %s ***", func.__qualname__)
        return result

    return wrapper


logger = logging.getLogger(__package__)
log = logger.log
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
