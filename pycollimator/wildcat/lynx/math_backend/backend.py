from functools import partial

import numpy as np

from .numpy import numpy_functions, numpy_constants
from .jax import jax_functions, jax_constants
from .casadi import casadi_functions, casadi_constants
from .torch import torch_functions, torch_constants


class MathBackend:
    _default_functions = [
        "dtype",
        "asarray",
        "array",
        "zeros",
        "zeros_like",
        "reshape",
        "cond",
        "sin",
    ]

    @classmethod
    def _raise_not_implemented(cls, name, *args, **kwargs):
        raise NotImplementedError(f"{name} not implemented for {cls.__name__}")

    @classmethod
    def _add_backend_function(cls, name):
        setattr(cls, name, partial(cls._raise_not_implemented, name))

    def __init__(self):
        for func in self._default_functions:
            if not hasattr(self, func):
                setattr(self, func, partial(self._raise_not_implemented, func))


def _make_backend(name, functions, constants):
    static_functions = {
        name: staticmethod(function) for name, function in functions.items()
    }
    attrs = {**static_functions, **constants}
    return type(name, (MathBackend,), attrs)()


class MathDispatcher:
    """Singleton class for calling out to the appropriate backend."""

    _active_backend = "jax"

    _backends = {
        "jax": _make_backend("JaxBackend", jax_functions, jax_constants),
        "torch": _make_backend("TorchBackend", torch_functions, torch_constants),
        "numpy": _make_backend("NumpyBackend", numpy_functions, numpy_constants),
        "casadi": _make_backend("CasadiBackend", casadi_functions, casadi_constants),
    }

    @property
    def active_backend(self):
        return self._active_backend

    def set_backend(self, backend):
        self._active_backend = backend

    def __getattr__(self, name):
        return getattr(self._backends[self.active_backend], name)

    def function(self, name):
        # These seem to have to be wrapped in a function to avoid fixing
        # the backend at the time of definition.
        def _call(*args, **kwargs):
            return getattr(self._backends[self.active_backend], name)(*args, **kwargs)

        return _call


math_backend = MathDispatcher()

# TODO: Do we need to be specific about which version of these constants gets used?
inf = np.inf
nan = np.nan

# Alias some core functions for convenience
dtype = math_backend.function("dtype")
asarray = math_backend.function("asarray")
array = math_backend.function("array")
zeros = math_backend.function("zeros")
zeros_like = math_backend.function("zeros_like")
reshape = math_backend.function("reshape")
cond = math_backend.function("cond")
