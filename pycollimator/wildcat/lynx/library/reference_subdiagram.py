from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from lynx import logging
from lynx.framework import InstanceParameter
from lynx.cli.types import ParameterDefinition


if TYPE_CHECKING:
    from lynx.framework import Diagram


def eval_parameter(name: str, value: Any, call_site_namespace: dict):
    p = eval(str(value), globals(), call_site_namespace)
    # Rules for user-input parameters:
    # - if explicitly given as an array with dtype, use that dtype
    # - if boolean, use as given
    # - otherwise, convert to a numpy array
    if not hasattr(p, "dtype") and not isinstance(p, bool):
        p = np.array(p)
        # if the array has integer dtype convert to float. Note that
        # this case will still not be reached if the parameter is explicitly
        # declared as an array with integer datatype.  However, this will
        # promote inputs like `"0"` or `"[1, 0]"` to floats.
        if issubclass(p.dtype.type, np.integer):
            p = p.astype(float)
    return InstanceParameter(name=name, string_value=str(value), evaluated_value=p)


class ReferenceSubdiagram:
    _registry: dict[str, Callable[[Any], "Diagram"]] = {}
    _parameter_definitions: dict[str, list[ParameterDefinition]] = {}

    @staticmethod
    def _check_params(ref_id: str, params: dict[str, Any]):
        if not params:
            return
        default_parameters = ReferenceSubdiagram.get_parameter_definitions(ref_id)
        default_params_names = set()
        if default_parameters:
            default_params_names = set([p.name for p in default_parameters])
        missing_defs = set(params.keys()) - default_params_names
        if missing_defs:
            logging.warning(
                "The following parameters were found in block but not "
                "defined in submodel %s:\n%s\n"
                "These parameter may not work inside new instances of this submodel.",
                ref_id,
                missing_defs,
            )

    @classmethod
    def create_diagram(
        cls,
        ref_id: str,
        *args,
        call_site_namespace: dict[str, Any] = None,
        instance_parameters: dict[str, Any] = None,
        **kwargs,
    ) -> "Diagram":
        """
        Create a diagram based on the given reference ID and parameters.

        Note that for submodels we evaluate all parameters, there is no
        "pure" string parameters.

        Args:
            ref_id (str): The reference ID of the diagram.
            *args: Variable length arguments.
            call_site_namespace (dict[str, Any], optional): The namespace of the call site. Defaults to None.
            instance_parameters (dict[str, Any], optional): The instance parameters for the diagram. Defaults to None.
                example: {"gain": 3.0}
            **kwargs: Keyword arguments.

        Returns:
            Diagram: The created diagram.

        Raises:
            ValueError: If the reference subdiagram with the given ref_id is not found.
        """
        if ref_id not in ReferenceSubdiagram._registry:
            raise ValueError(f"ReferenceSubdiagram with ref_id {ref_id} not found.")
        ReferenceSubdiagram._check_params(ref_id, instance_parameters)

        params_def = ReferenceSubdiagram.get_parameter_definitions(ref_id)
        params: dict[str, InstanceParameter] = {}
        if params_def:
            params = {
                p.name: eval_parameter(p.name, p.default_value, call_site_namespace)
                for p in params_def
                if not instance_parameters or p.name not in instance_parameters
            }

        new_instance_parameters: dict[str, InstanceParameter] = {}
        if instance_parameters:
            new_instance_parameters = {
                k: eval_parameter(k, v, call_site_namespace)
                for k, v in instance_parameters.items()
            }

            for name, param in new_instance_parameters.items():
                if name not in params:
                    params[name] = param
                else:
                    params[name].evaluated_value = param.evaluated_value

        diagram = ReferenceSubdiagram._registry[ref_id](
            *args, parameters=params, **kwargs
        )

        diagram.ref_id = ref_id
        diagram.instance_parameters = new_instance_parameters
        return diagram

    @staticmethod
    def register(
        ref_id: str,
        constructor: Callable[[Any], "Diagram"],
        parameter_definitions: list[ParameterDefinition] = None,
    ):
        logging.debug("Registering ReferenceSubdiagram with ref_id %s", ref_id)
        if ref_id in ReferenceSubdiagram._registry:
            logging.warning(
                "ReferenceSubdiagram with ref_id %s already registered.",
                ref_id,
            )
        ReferenceSubdiagram._registry[ref_id] = constructor
        ReferenceSubdiagram._parameter_definitions[ref_id] = parameter_definitions

    @staticmethod
    def get_parameter_definitions(ref_id: str) -> list[ParameterDefinition]:
        if ref_id not in ReferenceSubdiagram._parameter_definitions:
            raise ValueError(f"ReferenceSubdiagram with ref_id {ref_id} not found.")
        return ReferenceSubdiagram._parameter_definitions[ref_id]
