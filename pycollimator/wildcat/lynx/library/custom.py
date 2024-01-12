from __future__ import annotations

import math
from typing import TYPE_CHECKING, Hashable, List, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from lynx.framework.state import LeafState

from ..framework import LeafSystem
from ..logging import logger
from ..framework.error import BlockRuntimeError


if TYPE_CHECKING:
    from ..math_backend.typing import Array, DTypeLike, ShapeLike


__all__ = [
    "CustomJaxBlock",
    "CustomPythonBlock",
    "PythonScriptError",
]


class PythonScriptError(BlockRuntimeError):
    def __init__(self, block_id, message: str):
        super().__init__(block_id=block_id, message=message)


class CustomJaxBlock(LeafSystem):
    """Analogue to the CMLC PythonScript block but implemented using JAX.

    A few important notes and changes/limitations to this JAX implementation:
    - "Agnostic" mode is not supported (will be moved to CustomFeedthroughSystem)
    - For now we're assuming that all code is written using the JAX-supported subset
        of Python:
        * Numerical operations should use `jax.numpy = jnp` instead of `numpy = np`
        * Standard control flow is not supported (if/else, for, while, etc.). Instead
            use `lax.cond`, `lax.fori_loop`, `lax.while_loop`, etc.
            https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#structured-control-flow-primitives
        * Functions must be pure and arrays treated as immutable.
            https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates
        Provided these assumptions hold, the code can be JIT compiled, differentiated,
        run on GPU, etc.
    - As a consequence of the JAX-only assumption, arbitrary Python code is not yet
        supported. We can add a similar block that supports arbitrary Python code but
        prohibits JAX tracing (or modify this block to support both cases).
    - Variable scoping works slightly differently than in the CMLC PythonScript.
        Specifically, the `init_code` and `step_code` are executed in the same scope,
        so variables declared in the `init_code` will be available in the `step_code`
        and can be modified in that scope. Internally, everything declared in
        `init_code` is treated as a single discrete state.
        However, variables declared in the `step_code` will NOT persist between
        evaluations. Users should think of `step_code` as a normal Python function
        where locally declared variables will disappear on leaving the scope.
    - Persistent variables (outputs and anything declared in `init_code`) must have
        static shapes and dtypes. This means that you cannot declare `x = 0.0` in
        `init_code` and then later assign `x = jnp.zeros(4)` in `step_code`.

    These changes mean that many PythonScript blocks will not be backwards compatible.

    For example, the "relay_psb" block in the CMLC test case had no `init_code` and the
    following `step_code`:
    ```
    if time ==0:
        state = 0.0

    if in_0 > 0.5:
        state = 1.0
    elif in_0 < -0.5:
        state = 0.0
    else:
        pass
    out_0 = state
    ```

    In the JAX implementation, the `state` variable needs to be declared in the `init_code`:
    ```
    state = 0.0
    ```

    The `step_code` can use `jnp.where` instead of `if/else`:
    ```
    state = jnp.where(in_0 > 0.5, 1.0, state)
    state = jnp.where(in_0 < -0.5, 0.0, state)
    out_0 = state
    ```
    """

    def __init__(
        self,
        dt: float,
        init_script: str = "",
        user_statements: str = "",
        finalize_script: str = "",  # presently ignored for JAX block
        inputs: List[str] = None,  # [name]
        outputs: Mapping[str, Tuple[DTypeLike, ShapeLike]] = None,
        name: str = None,
        system_id: Hashable = None,
        **parameters: Mapping[str, Array],
    ):
        super().__init__(name=name, system_id=system_id)

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = {}

        self.dt = dt
        self.init_code = init_script
        self.step_code = user_statements
        self.finalize_code = finalize_script
        if finalize_script != "" and not isinstance(self, CustomPythonBlock):
            raise NotImplementedError(
                f"PythonScript block '{name}' has finalize_script but this is not supported."
            )

        # Declare an input port for each of the input variables
        self.input_names = inputs
        for name in inputs:
            self.declare_input_port(name)

        # Declare an output port and discrete state for each of the output variables
        self.outputs = {}
        self.output_port_params = {}  # Used for serialization/deserialization
        for o_port_name, (dtype, shape) in outputs.items():
            self.outputs[o_port_name] = self.declare_discrete_state(
                shape=shape, dtype=dtype
            )
            self.output_port_params[o_port_name] = {"dtype": dtype, "shape": shape}
            self.declare_discrete_state_output(
                name=o_port_name, state_index=self.outputs[o_port_name]
            )

        # Declare parameters
        for param_name, value in parameters.items():
            if isinstance(value, list):
                value = jnp.asarray(value)
            as_array = isinstance(value, jnp.ndarray) or jnp.isscalar(value)
            self.declare_parameter(param_name, value, as_array=as_array)

        # Run the init_script
        self.exec_init()

        # The step function acts as a periodic update that will update all components
        # of the discrete state.
        self.declare_periodic_unrestricted_update(
            self.exec_step,
            period=dt,
            offset=dt,
        )

        self.declare_configuration_parameters(
            dt=dt,
            init_script=init_script,
            user_statements=user_statements,
            finalize_script=finalize_script,
        )

    @property
    def local_env_base(self):
        # Define a starting point for the local code execution environment.
        # we have to inclide __main__ so that the code behaves like a module.
        # this allows for code like this:
        #   imports ...
        #   a = 1
        #   def f(b):
        #       return a+b
        #   out_0 = f(2)
        #
        # with getting a 'a not defined' error.
        return {
            "__main__": {},
            "lax": lax,
            "jnp": jnp,
            "jax": jax,
        }

    def filter_globals(self, globals_dict):
        """
        this function is a bit hacky (got it form ChatGPT).
        the poit is that since we have to use locals as globals,
        our locals gets really poluted with all kinds of stuff.
        trying to retain this whole thing results in jax errors.
        so this functions job is to discard anything that jax cannot trace.
        """
        filtered_globals = {}
        for key, value in globals_dict.items():
            if isinstance(value, (int, float, complex, np.ndarray)):
                filtered_globals[key] = value
        return filtered_globals

    def exec_init(self) -> None:
        # Before executing the step code, we have to build up the local environment.
        # This includes specified modules, python block user defined parameters.
        local_env = {
            **self.local_env_base,
            **self._default_parameters,
        }

        # similar to above where we included __main__ so the code behaves as a module,
        # here we have to pass the local_env with __main__ as 1] globals, since that
        # is what allow the code to be executed as a module. 2] local since that is where
        # the new binding will be written, that we need to retain sicne the code in step_code
        # may depend on these bindings.
        exec(self.init_code, local_env, local_env)

        # persistent_env contains bindings for parameters and for values from init_script
        persistent_env = self.filter_globals(local_env)

        # Store the dictionary of local environment variables as a discrete state
        # This is the only persistent state of the system - anything declared in
        # the "step" function will be forgotten at the end of the step
        self.local_env_index = self.declare_discrete_state(
            default_value=persistent_env, as_array=False
        )

    def exec_step(self, time, state: LeafState, *inputs, **parameters):
        # Before executing the step code, we have to build up the local environment.
        # This includes the persistent variables (anything declared in `init_code`),
        # time, block inputs, user-defined parameters, and specified modules.

        # Retrieve the variables declared in `init_code` from the discrete state
        persistent_env = state.discrete_state[self.local_env_index]

        # Inputs are in order of port declaration, so they match `self.input_names`
        input_env = dict(zip(self.input_names, inputs))

        # Create a dictionary of all the information that the step function will need
        base_copy = self.local_env_base.copy()
        local_env = {
            "time": time,
            **base_copy,
            **persistent_env,
            **input_env,
        }

        # Execute the step code in the local environment
        # print(f"{self.name} local_env=\n{local_env.keys()}")
        exec(self.step_code, local_env, local_env)

        xd = state.discrete_state.copy()

        # Store the persistent variables in the corresponding discrete state
        xd[self.local_env_index] = {key: local_env[key] for key in persistent_env}

        # Store the output variables in the corresponding discrete state
        for name, index in self.outputs.items():
            xd[index] = local_env[name]

        return state.with_discrete_state(xd)


class CustomPythonBlock(CustomJaxBlock):
    """Container for arbitrary user-defined Python code.

    Implemented to support legacy PythonScript blocks (Discrete mode only)

    Not traceable (no JIT compilation or autodiff). The internal implementation
    and behavior of this block differs vastly from the JAX-compatible block as
    this block stores state directly within the Python instance. Objects
    and modules can be kept as discrete state.
    """

    enable_trace_cache_sources = False
    enable_trace_discrete_updates = False
    enable_trace_unrestricted_updates = False
    enable_trace_time_derivatives = False

    def __init__(
        self,
        dt: float,
        init_script: str = "",
        user_statements: str = "",
        finalize_script: str = "",  # presently ignored
        inputs: List[str] = None,  # [name]
        outputs: Mapping[str, Tuple[DTypeLike, ShapeLike]] = None,
        name: str = None,
        system_id: Hashable = None,
        **parameters: Mapping[str, Array],
    ):
        self.persistent_env = None
        super().__init__(
            dt=dt,
            init_script=init_script,
            user_statements=user_statements,
            finalize_script=finalize_script,
            inputs=inputs,
            outputs=outputs,
            name=name,
            system_id=system_id,
            **parameters,
        )

    @property
    def local_env_base(self):
        # Define a starting point for the local code execution environment.
        # For now this is just numpy, but we could add other modules as well.
        # @jp FIXME? We probably should just have a clean env here. Importing
        # math, np and such should be done in the user code.
        return {
            "__main__": {},
            "np": np,
            "math": math,
            "numpy": np,
            "true": True,
            "false": False,
        }

    def filter_globals(self, globals_dict):
        # See https://collimator.atlassian.net/browse/WC-98
        # The objective of this function is simply to make both python script
        # blocks a little bit more similar (JAX and non-JAX).
        """
        Keep only the persistent variables in the global environment,
        that is, those that have been initialized in the 'init_script' code.
        """
        return {k: v for k, v in globals_dict.items() if k in self.persistent_env}

    def exec_init(self) -> None:
        # State will be stored in the Python instance, outside of JAX
        local_env = {
            **self.local_env_base,
            **self._default_parameters,
        }
        try:
            exec(self.init_code, local_env, local_env)
        except KeyboardInterrupt:
            logger.error(
                "Python block '%s' init script execution was interrupted.", self.name
            )
            raise
        except Exception as e:
            raise PythonScriptError(self.name, str(e)) from e
        self.persistent_env = local_env

    def exec_step(self, time, state, *inputs, **parameters):
        logger.debug(f"Executing step for {self.name} with {state=}, {inputs=}")
        # Inputs are in order of port declaration, so they match `self.input_names`
        input_env = dict(zip(self.input_names, inputs))

        base_copy = self.local_env_base.copy()
        local_env = {
            "time": time,
            **base_copy,
            **(self.persistent_env or {}),
            **input_env,
        }

        # Execute the step code in the local environment
        try:
            exec(self.step_code, local_env, local_env)
        except KeyboardInterrupt:
            logger.error(
                "Python block '%s' init script execution was interrupted.", self.name
            )
            raise
        except NameError as e:
            err_msg = (
                f"Python block '{self.name}' step script execution failed with a NameError on"
                + f" missing variable '{e.name}'."
                + " All names used in this script should be declared in the init script."
                + f" The execution environment contains the following names: {', '.join(list(local_env.keys()))}"
            )
            logger.error(err_msg)
            logger.error("NameError: %s", e)
            raise PythonScriptError(self.name, err_msg) from e
        except Exception as e:
            raise PythonScriptError(self.name, str(e)) from e

        # Keep local env for next step but only if defined in init_script
        # NOTE: If this restriction turns out to be counterproductive, we can
        # remove it and remove the NameError handling above as well. The thinking
        # here is that this could help avoiding stuff like `if time == 0: x = 0`
        self.persistent_env = self.filter_globals(local_env)

        # Store the output variables in the corresponding discrete state
        xd = state.discrete_state.copy()
        for name, index in self.outputs.items():
            xd[index] = local_env[name]
        return state.with_discrete_state(xd)

    def exec_finalize(self):
        if self.finalize_code is None or len(self.finalize_code) == 0:
            return

        # Not a step => no inputs
        local_env = {
            **self.local_env_base,
            **(self.persistent_env or {}),
        }

        # Execute the step code in the local environment
        exec(self.finalize_code, local_env, local_env)
