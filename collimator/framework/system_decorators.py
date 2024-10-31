# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

from functools import wraps
import inspect
from typing import (
    Callable,
    ParamSpec,
    TypeVar,
    Union,
)
from .parameter import Parameter


__all__ = ["parameters", "ports"]


def _get_arg_names(func):
    sig = inspect.signature(func)
    positional_arg_names = []
    all_args = []
    for name, param in sig.parameters.items():
        is_positional_or_keyword = param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )
        if is_positional_or_keyword and param.default is inspect.Parameter.empty:
            positional_arg_names.append(name)
        all_args.append(name)

    return all_args, positional_arg_names


def _get_default_value(init_func, param_name):
    sig = inspect.signature(init_func)
    param = sig.parameters[param_name]
    if param.default is inspect.Parameter.empty:
        return None
    return param.default


def _get_params(param_names, init_func, args, kwargs, use_defaults=True):
    all_args, args_names = _get_arg_names(init_func)
    all_args = all_args[1:]  # remove self
    args_names = args_names[1:]  # remove self

    params = {}
    for k in param_names:
        if k in kwargs:
            params[k] = kwargs[k]
        elif k in args_names:
            idx = args_names.index(k)
            if idx >= len(args):
                raise ValueError(f"Missing required argument {k}")
            params[k] = args[idx]
        elif k in all_args and all_args.index(k) < len(args):
            params[k] = args[all_args.index(k)]
        elif use_defaults:
            # kwarg's default value
            params[k] = _get_default_value(init_func, k)
    return params


P = ParamSpec("P")
T = TypeVar("T")


# TODO: deprecate decorator at function level?
def parameters(static: list[str] = None, dynamic: list[str] = None):
    """Decorator to apply to a system class to declare
    static or dynamic parameters."""

    if static is None:
        static = []

    if dynamic is None:
        dynamic = []

    static_param_names = set(static)
    dynamic_param_names = set(dynamic)

    def decorator(entity: Union[Callable[P, T], type]) -> Callable[P, T]:
        if isinstance(entity, type):
            init_func = entity.__init__
            # Useful for class introspection like parsing custom leaf system in
            # the frontend to configure the UI block.
            entity.__parameters__ = static + dynamic
        elif callable(entity):
            init_func = entity

        @wraps(init_func)
        def wrapped_init(self, *args, **kwargs):
            resolved_args = [
                arg.get() if isinstance(arg, Parameter) else arg for arg in args
            ]
            resolved_kwargs = {
                k: kwarg.get() if isinstance(kwarg, Parameter) else kwarg
                for k, kwarg in kwargs.items()
            }

            init_func(self, *resolved_args, **resolved_kwargs)

            # TODO: Prevent parameters from being inherited from parent systems.
            # This is necessary to avoid unknown behaviors when a child parameter
            # is used to define a parent parameter, eg. what we used to do in
            # PID continuous block where gains were used to calculate A, B, C, D
            # matrices.
            # This will force the implementor of the block to implement jitted
            # callbacks in such a way that they only depend on the current system's
            # parameters.
            # We should also allow inheritance of params with a flag or annotation.
            # https://github.com/collimator-ai/collimator/pull/6790
            # self._static_parameters = {}
            # self._dynamic_parameters = {}

            static_params = _get_params(static_param_names, init_func, args, kwargs)
            for param_name, value in static_params.items():
                self.declare_static_parameter(param_name, value)

            dyn_params = _get_params(dynamic_param_names, init_func, args, kwargs)
            for param_name, value in dyn_params.items():
                if value is not None:
                    self.declare_dynamic_parameter(param_name, value)

        if isinstance(entity, type):
            entity.__init__ = wrapped_init
            return entity
        elif callable(entity):
            return wrapped_init

    return decorator


def ports(inputs: Union[list[str], int], outputs: Union[list[str], int]):
    def decorator(cls: type) -> type:
        original_init = cls.__init__
        cls.__input_ports__ = inputs
        cls.__output_ports__ = outputs

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            if isinstance(inputs, list):
                for name in inputs:
                    self.declare_input_port(name=name)
            elif isinstance(inputs, int):
                for _ in range(inputs):
                    self.declare_input_port()
            else:
                raise ValueError("input must be list of names or int")

            if isinstance(outputs, list):
                for name in outputs:
                    self.declare_output_port(None, name=name)
            elif isinstance(outputs, int):
                for _ in range(outputs):
                    self.declare_output_port(None)
            else:
                raise ValueError("input must be list of names or int")

        cls.__init__ = new_init
        return cls

    return decorator
