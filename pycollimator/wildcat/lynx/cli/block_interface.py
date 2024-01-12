import jax.numpy as jnp
import numpy as np

from .. import library
from lynx.cli.types import Node

"""
All of the <BlahBlock> functions should eventually be ported to
their respective blocks class definition, e.g. in library.primitives.py
"""


def PythonScriptBlock(
    block_spec: Node,
    io_ports_params: dict,
    accelerate_with_jax=True,
    dt: str = None,
    discrete_interval: float = None,
    **kwargs,
):
    inputs_ = [d.name for d in block_spec.inputs]
    outputs_ = {}

    for output in block_spec.outputs:
        output_params = io_ports_params[output.name]
        dtype = output_params["dtype"]
        shape = output_params["shape"]

        if isinstance(shape, (int, float)):
            shape = tuple([int(shape)])
        elif isinstance(shape, np.ndarray) and shape.shape == ():
            shape = tuple([int(shape)])
        else:
            shape_list = [int(el) for el in shape]
            shape = tuple(shape_list)

        outputs_[output.name] = (dtype, shape)

    block_cls = (
        library.CustomJaxBlock if accelerate_with_jax else library.CustomPythonBlock
    )
    return _wrap(block_cls)(
        inputs=inputs_,
        outputs=outputs_,
        dt=float(dt) if dt is not None else discrete_interval,
        **kwargs,
    )


def IntegratorBlock(initial_states, **kwargs):
    return _wrap(library.Integrator)(
        initial_continuous_state=initial_states,
        **kwargs,
    )


def CosineWaveBlock(**kwargs):
    kwargs["phase"] = jnp.pi / 2 + kwargs["phase"]
    return _wrap(library.Sine)(
        **kwargs,
    )


def ClockBlock(block_spec, discrete_interval, **kwargs):
    if (
        not block_spec.time_mode
        or block_spec.time_mode == "agnostic"
        or block_spec.time_mode
        == "continuous"  # FIXME: had to put this here because serialization tests fail.
    ):
        return _wrap(library.Clock)(**kwargs)
    elif block_spec.time_mode == "discrete":
        return _wrap(library.DiscreteClock)(discrete_interval, **kwargs)
    else:
        raise ValueError(
            f"Clock block got unrecognized time_mode: {block_spec.time_mode}"
        )


def DemuxBlock(block_spec, **kwargs):
    n_out = len(block_spec.outputs)
    return _wrap(library.Demultiplexer)(
        n_out,
        **kwargs,
    )


def StackBlock(block_spec, axis, **kwargs):
    n_in = len(block_spec.inputs)
    return _wrap(library.Stack)(
        n_in,
        axis,
        **kwargs,
    )


def TransferFunctionBlock(numerator_coefficients, denominator_coefficients, **kwargs):
    return _wrap(library.TransferFunction)(
        numerator_coefficients,
        denominator_coefficients,
        **kwargs,
    )


def StateSpaceBlock(A, B, C, D, initial_states=None, **kwargs):
    return _wrap(library.LTISystem)(
        np.array(A),
        np.array(B),
        np.array(C),
        np.array(D),
        initialize_states=initial_states,
        **kwargs,
    )


def ModelicaFMUBlock(
    file_name: str,
    discrete_interval: float,
    block_spec: Node,
    name: str = None,
    system_id: str = None,
    io_ports_params: dict = None,
    fmu_guid: str = None,
    **kwargs,
):
    input_names = [d.name for d in block_spec.inputs]
    output_names = [d.name for d in block_spec.outputs]

    return _wrap(library.ModelicaFMU)(
        file_name=file_name,
        dt=discrete_interval,
        name=name,
        system_id=system_id,
        input_names=input_names,
        output_names=output_names,
        parameters=kwargs,
    )


def IntegratorDiscreteBlock(discrete_interval, initial_states, **kwargs):
    # some old models still have these params in the json, eventhough
    # they are not in the schemas.
    # https://www.notion.so/Wildcat-legacy-support-workarounds-ad3f8141906c4665992e2a2e93cb6270?pvs=4#5075d9f8655e4305b3738d72409eaed5
    if "hold_trigger_method" in kwargs:
        kwargs.pop("hold_trigger_method")
    if "reset_trigger_method" in kwargs:
        kwargs.pop("reset_trigger_method")
    return _wrap(library.IntegratorDiscrete)(
        dt=discrete_interval,
        initial_state=initial_states,
        **kwargs,
    )


def DiscretePIDBlock(discrete_interval, Kp, Ki, Kd, **kwargs):
    dt = kwargs.pop("dt", None)
    return _wrap(library.DiscretePID)(
        kp=Kp,
        ki=Ki,
        kd=Kd,
        dt=dt if dt is not None else discrete_interval,
        **kwargs,
    )


def PIDBlock(Kp, Ki, Kd, N=100, **kwargs):
    return _wrap(library.PID)(
        kp=Kp,
        ki=Ki,
        kd=Kd,
        n=N,
        **kwargs,
    )


def MLPBlock(file_name, model_format, **kwargs):
    return _wrap(library.MLP)(
        filename=file_name,
        **kwargs,
    )


def DerivativeBlock(N=100, **kwargs):
    return _wrap(library.Derivative)(
        N,
        **kwargs,
    )


def IOPortBlock(**kwargs):
    if "description" in kwargs:
        kwargs.pop("description")
    if "port_id" in kwargs:
        kwargs.pop("port_id")
    if "default_value" in kwargs:
        kwargs.pop("default_value")
    return _wrap(library.IOPort)(
        **kwargs,
    )


def _convert_bools(params: dict):
    bools = {
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }
    return {
        k: v
        if type(v) is not str or v not in ("true", "false", "True", "False")
        else bools[v]
        for k, v in params.items()
    }


def _wrap(block_cls):
    def _wrapped(
        *args, io_ports_params=None, block_spec=None, discrete_interval=None, **kwargs
    ):
        return block_cls(*args, **_convert_bools(kwargs))

    return _wrapped


def _wrap_reducer(block_cls):
    def _wrapped(
        *args, io_ports_params=None, block_spec=None, discrete_interval=None, **kwargs
    ):
        n_in = len(block_spec.inputs)
        return block_cls(n_in, **_convert_bools(kwargs))

    return _wrapped


def _wrap_discrete(block_cls):
    def _wrapped(
        *args, io_ports_params=None, block_spec=None, discrete_interval=None, **kwargs
    ):
        return block_cls(discrete_interval, **_convert_bools(kwargs))

    return _wrapped


def get_block_fcn(node_type: str = "core.Adder"):
    fcn_map = {
        "core.Abs": _wrap(library.Abs),
        "core.Adder": _wrap_reducer(library.Adder),
        # bus creator
        # bus selector
        # c function
        "core.Chirp": _wrap(library.Chirp),
        "core.Clock": ClockBlock,
        "core.Comparator": _wrap(library.Comparator),
        # conditional
        "core.Constant": _wrap(library.Constant),
        # "core.CoordinateRotation": _wrap(library.None),
        # coordinate rotation conversion
        "core.CosineWave": CosineWaveBlock,
        # cpp function
        "core.CrossProduct": _wrap(library.CrossProduct),
        "core.DataSource": _wrap(library.DataSource),
        "core.DeadZone": _wrap(library.DeadZone),
        # "core.Delay": _wrap(library.None),
        "core.Demux": DemuxBlock,
        "core.Derivative": DerivativeBlock,
        "core.DerivativeDiscrete": _wrap_discrete(library.DerivativeDiscrete),
        # discrete initializer
        "core.DotProduct": _wrap(library.DotProduct),
        # drive cycle
        "core.EdgeDetection": _wrap_discrete(library.EdgeDetection),
        "core.Exponent": _wrap(library.Exponent),
        # filter discrete
        "core.Gain": _wrap(library.Gain),
        # group implemented in model_interface.py
        "core.IfThenElse": _wrap(library.IfThenElse),
        # image segmentation
        # image source
        "core.Inport": IOPortBlock,
        "core.Integrator": IntegratorBlock,
        "core.IntegratorDiscrete": IntegratorDiscreteBlock,
        # iterator [and its loop control blocks]
        # linaerized sysyem
        "core.LogicalOperator": _wrap_reducer(library.LogicalOperator),
        "core.LookupTable1d": _wrap(library.LookupTable1d),
        "core.LookupTable2d": _wrap(library.LookupTable2d),
        "core.Log": _wrap(library.Log),
        # matrix concat. @am. this should be depreciated in favor of Stack
        "core.MatrixInversion": _wrap(library.MatrixInversion),
        "core.MatrixMultiplication": _wrap_reducer(library.MatrixMultiplication),
        "core.MatrixTransposition": _wrap(library.MatrixTransposition),
        "core.MinMax": _wrap_reducer(library.MinMax),
        "core.ModelicaFMU": ModelicaFMUBlock,
        "core.Mux": _wrap_reducer(library.Multiplexer),
        # object detection
        "core.Offset": _wrap(library.Offset),
        "core.Outport": IOPortBlock,
        "core.PID": PIDBlock,
        "core.PID_Discrete": DiscretePIDBlock,
        "core.Power": _wrap(library.Power),
        "core.Predictor": MLPBlock,
        "core.Product": _wrap_reducer(library.Product),
        "core.ProductOfElements": _wrap(library.ProductOfElements),
        "core.Pulse": _wrap(library.Pulse),
        "core.PythonScript": PythonScriptBlock,
        "core.Quantizer": _wrap(library.Quantizer),
        "core.Ramp": _wrap(library.Ramp),
        # random normal
        # "core.RateLimiter": _wrap(library.None),
        "core.Reciprocal": _wrap(library.Reciprocal),
        # "core.ReferenceSubmodel": implemented in model_interface.py
        # ("core.RigidBody",RigidBody.defnlibrary.);
        "core.Saturate": _wrap(library.Saturate),
        "core.Sawtooth": _wrap(library.Sawtooth),
        "core.ScalarBroadcast": _wrap(library.ScalarBroadcast),
        # ("core.SignalDatatypeConversion",SignalDatatypeConversion.defnlibrary.);
        "core.SineWave": _wrap(library.Sine),
        "core.SINDy": _wrap(library.ContinuousTimeSindyWithControl),
        "core.Slice": _wrap(library.Slice),
        "core.SquareRoot": _wrap(library.SquareRoot),
        "core.Stack": StackBlock,
        # state machine
        "core.StateSpace": StateSpaceBlock,
        "core.Step": _wrap(library.Step),
        # ("core.Stop",Stop.defnlibrary.);
        "core.SumOfElements": _wrap(library.SumOfElements),
        "core.TransferFunction": TransferFunctionBlock,
        # ("core.TransferFunctionDiscrete",TransferFunctionDiscrete.defnlibrary.);
        "core.Trigonometric": _wrap(library.Trigonometric),
        "core.UnitDelay": _wrap_discrete(library.UnitDelay),
        # video sink
        # video source
        # ("core.WhiteNoise",WhiteNoise.defnlibrary.);
        "core.ZeroOrderHold": _wrap_discrete(library.ZeroOrderHold),
    }

    return fcn_map[node_type]
