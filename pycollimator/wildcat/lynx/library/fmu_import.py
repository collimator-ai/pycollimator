"""
basic FMU import block based on FMPy package and the example:
https://github.com/CATIA-Systems/FMPy/blob/main/fmpy/examples/custom_input.py
"""

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave, FMICallException
from fmpy.model_description import ScalarVariable

from ..framework.error import BlockRuntimeError
from ..framework import LeafSystem, BlockInitializationError
from ..logging import logger


ValueReference = int


class ModelicaFMU(LeafSystem):
    enable_trace_cache_sources = False
    enable_trace_discrete_updates = False
    enable_trace_unrestricted_updates = False
    enable_trace_time_derivatives = False

    # Should we pass parameter overrides via kwargs? Sounds like it could conflict
    # in some rare cases (eg. dt, name...). The corresponding definition in
    # block_interface.py is pretty fragile in this regard.
    def __init__(
        self,
        file_name,
        dt,
        name=None,
        system_id=None,
        input_names: list[str] = None,
        output_names: list[str] = None,
        parameters: dict = None,
        **kwargs,
    ):
        """Import FMU into Lynx framework.

        Args:
            file_name (str): path to FMU file
            dt (float): stepsize for FMU simulation
            name (str, optional): name of block
            system_id (str, optional): system id of block
            input_names (list[str], optional): if set, only expose these inputs
            output_names (list[str], optional): if set, only expose these outputs
            parameters (dict, optional): dictionary of parameter overrides
            kwargs: ignored
        """
        try:
            super().__init__(name=name, system_id=system_id)
            self._init(
                file_name,
                dt,
                name=name,
                system_id=system_id,
                input_names=input_names,
                output_names=output_names,
                parameters=parameters,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize FMU block %s (%s): %s", name, system_id, e
            )
            raise BlockInitializationError(system_id or name, str(e))

    def _init(
        self,
        file_name,
        dt,
        name=None,
        system_id=None,
        input_names: list[str] = None,
        output_names: list[str] = None,
        parameters: dict = None,
    ):
        self.declare_configuration_parameters(file_name=file_name)

        self.dt = dt

        # read the model description
        model_description = read_model_description(file_name)

        # extract the FMU
        unzipdir = extract(file_name)

        self.fmu = fmu = FMU2Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName=system_id or name,
        )

        # initialize
        fmu.instantiate()

        # FIXME: can we set this to the user specificed startTime?
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()

        # collect the value references
        self.fmu_inputs: list[ValueReference] = []
        self.fmu_outputs: list[ValueReference] = []

        inputs_by_name: dict[str, ScalarVariable] = {}
        outputs_by_name: dict[str, ScalarVariable] = {}
        variable_by_id: dict[int, ScalarVariable] = {}

        # FIXME: we rely on the XML file here, but collimator uses a similar
        # JSON file with altered variable names.
        # TODO: implement support for parsing that file and mapping from
        # collimator json name to/from xml name properly.
        def _compatible_param_name(name):
            return name.replace(".", "_")

        for variable in model_description.modelVariables:
            if variable.causality == "input":
                variable_by_id[variable.valueReference] = variable
                inputs_by_name[variable.name] = variable
            elif variable.causality == "output":
                variable_by_id[variable.valueReference] = variable
                outputs_by_name[variable.name] = variable
            elif variable.causality == "parameter" and parameters is not None:
                compat_name = _compatible_param_name(variable.name)
                parameter_value = parameters.get(compat_name, None)
                if parameter_value is None:
                    continue

                logger.debug(
                    "Setting parameter #%d '%s' <%s>: %s %s",
                    variable.valueReference,
                    variable.name,
                    variable.type,
                    parameter_value,
                    type(parameter_value),
                )

                # Values at this point have been wrapped into np.ndarray of
                # shape () via wildcat's JSON parsing. Enumerations are ints.
                match variable.type:
                    case "Boolean":
                        parameter_value = bool(parameter_value)
                        fmu.setBoolean([variable.valueReference], [parameter_value])
                    case "Integer":
                        parameter_value = int(parameter_value)
                        fmu.setInteger([variable.valueReference], [parameter_value])
                    case "Real":
                        parameter_value = float(parameter_value)
                        fmu.setReal([variable.valueReference], [parameter_value])
                    case "String":
                        parameter_value = str(parameter_value)
                        fmu.setString([variable.valueReference], [parameter_value])
                    case "Enumeration":
                        parameter_value = int(parameter_value)
                        fmu.setInteger([variable.valueReference], [parameter_value])
                    case _:
                        raise NotImplementedError(
                            f"Unsupported type for parameter {variable.name} in "
                            + f"FMU block {system_id or name}: {variable.type}"
                        )

        # If input_names or output_names are set, we filter out the variables
        # exposed as I/O ports to match those. This so that the ports in model.json
        # actually match those in the FMU.
        # NOTE: Maybe this is unnecessarily complicated.
        if input_names is not None:
            for name in input_names:
                if name not in inputs_by_name:
                    raise ValueError(
                        f"Input port {name} found on the block {system_id or name} "
                        + f"but not found in FMU {file_name}"
                    )
                variable = inputs_by_name[name]
                self.fmu_inputs.append(variable.valueReference)
                self.declare_input_port(name=variable.name)
        else:
            for name, variable in inputs_by_name.items():
                self.fmu_inputs.append(variable.valueReference)
                self.declare_input_port(name=name)

        if output_names is not None:
            for name in output_names:
                if name not in outputs_by_name:
                    raise ValueError(
                        f"Input port {name} found on the block {system_id or name} "
                        + f"but not found in FMU {file_name}"
                    )
                variable = outputs_by_name[name]
                self.fmu_outputs.append(variable.valueReference)
        else:
            for name, variable in outputs_by_name.items():
                self.fmu_outputs.append(variable.valueReference)

        fmu.exitInitializationMode()

        for output_ref in self.fmu_outputs:
            variable = variable_by_id[output_ref]
            match variable.type:
                case "Boolean":
                    start_value = fmu.getBoolean([variable.valueReference])[0]
                case "Integer" | "Enumeration":
                    start_value = fmu.getInteger([variable.valueReference])[0]
                case "Real":
                    start_value = fmu.getReal([variable.valueReference])[0]
                case _:
                    raise NotImplementedError(
                        f"Unsupported type for output port {variable.name} in FMU: {variable.type}"
                    )
            ds_output = self.declare_discrete_state(default_value=start_value)
            self.declare_discrete_state_output(
                name=variable.name, state_index=ds_output
            )

        def _get_fmu(context):
            # Extract a reference to the cached FMU from the context
            fmu = context[self.system_id].cache[self.fmu_index].value
            return fmu

        # Since the state is reserved for things that are convertible to an array
        # (at some tree depth), we have to store the FMU in a cache entry.
        # The outputs will be stored in a discrete state that is updated periodically.
        self.fmu_index = self.declare_cache(_get_fmu, default_value=fmu, name="fmu")
        self.fmu_cache = self.cache_sources[self.fmu_index]

        # The step function acts as a periodic update that will update all components
        # of the discrete state.
        self.declare_periodic_unrestricted_update(
            self.exec_step,
            period=dt,
            offset=dt,
        )

    # Since we actually need the cache in this particular case, override the
    # default behavior of deleting the whole thing
    def clear_cache(self, cache):
        return {self.fmu_index: cache[self.fmu_index]}

    def exec_step(self, time, state, *inputs, **parameters):
        # NOTE: We should get the fmu from the context in order to build a pure
        # function but it is very unlikely this would ever work with FMUs since
        # they have their own internal hidden state. More context here:
        # https://github.com/collimator-ai/collimator/pull/5330/files#r1419062533
        # Also look at that PR to see the previous implementation (it worked with
        # a single I/O port).

        try:
            fmu = self.fmu

            # Set inputs
            fmu.setReal(self.fmu_inputs, list(inputs))

            # Advance the FMU in time
            fmu.doStep(currentCommunicationPoint=time, communicationStepSize=self.dt)

            # The outputs will be stored in the discrete state
            fmu_out = fmu.getReal(self.fmu_outputs)
            state = state.with_discrete_state(fmu_out)
        except FMICallException as e:
            logger.error(
                "Failed to run FMU block %s (%s): %s", self.name, self.system_id, e
            )
            raise BlockRuntimeError(self.system_id or self.name, str(e))

        return state
