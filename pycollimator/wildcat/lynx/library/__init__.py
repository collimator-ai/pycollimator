"""Wildcat block library"""

from .generic import (
    SourceBlock,
    FeedthroughBlock,
    ReduceBlock,
)
from .primitives import (
    Abs,
    Constant,
    Sine,
    Clock,
    Comparator,
    CrossProduct,
    DiscreteClock,
    EdgeDetection,
    Exponent,
    Gain,
    IfThenElse,
    Offset,
    Reciprocal,
    LogicalOperator,
    MatrixInversion,
    MatrixMultiplication,
    MatrixTransposition,
    MinMax,
    Multiplexer,
    Demultiplexer,
    DeadZone,
    DotProduct,
    Adder,
    Product,
    ProductOfElements,
    Power,
    IOPort,
    Log,
    LookupTable1d,
    LookupTable2d,
    Chirp,
    Pulse,
    Quantizer,
    Sawtooth,
    ScalarBroadcast,
    SumOfElements,
    Slice,
    Stack,
    Step,
    SquareRoot,
    Ramp,
    Saturate,
    DiscretePID,
    ZeroOrderHold,
    UnitDelay,
    DerivativeDiscrete,
    Integrator,
    IntegratorDiscrete,
    Trigonometric,
)
from .custom import (
    CustomJaxBlock,
    CustomPythonBlock,
)
from .wrappers import (
    ode_block,
    feedthrough_block,
)
from .linear_system import (
    LTISystem,
    TransferFunction,
    linearize,
    PID,
    Derivative,
)

from .mpc import (
    LinearDiscreteTimeMPC,
    LinearDiscreteTimeMPC_OSQP,
)

from .nn import (
    MLP,
    QuadraticCost,
)

from .fmu_import import (
    ModelicaFMU,
)

from .sindy import (
    ContinuousTimeSindyWithControl,
)

from .data_source import (
    DataSource,
)

from .reference_subdiagram import ReferenceSubdiagram

__all__ = [
    "SourceBlock",
    "FeedthroughBlock",
    "ReduceBlock",
    "Abs",
    "Constant",
    "Sine",
    "Clock",
    "Comparator",
    "CrossProduct",
    "CustomJaxBlock",
    "CustomPythonBlock",
    "DataSource",
    "DeadZone",
    "Derivative",
    "DerivativeDiscrete",
    "DotProduct",
    "DiscreteClock",
    "EdgeDetection",
    "Exponent",
    "ModelicaFMU",
    "Gain",
    "IfThenElse",
    "Offset",
    "Reciprocal",
    "LogicalOperator",
    "MatrixInversion",
    "MatrixMultiplication",
    "MatrixTransposition",
    "MinMax",
    "Multiplexer",
    "Demultiplexer",
    "Adder",
    "PID",
    "Product",
    "ProductOfElements",
    "Power",
    "Integrator",
    "IntegratorDiscrete",
    "IOPort",
    "Log",
    "LookupTable1d",
    "LookupTable2d",
    "Chirp",
    "Pulse",
    "Quantizer",
    "Sawtooth",
    "ScalarBroadcast",
    "ContinuousTimeSindyWithControl",
    "SumOfElements",
    "Slice",
    "Stack",
    "Step",
    "SquareRoot",
    "Ramp",
    "Saturate",
    "DiscretePID",
    "ZeroOrderHold",
    "UnitDelay",
    "ode_block",
    "feedthrough_block",
    "LTISystem",
    "TransferFunction",
    "linearize",
    "LinearDiscreteTimeMPC",
    "LinearDiscreteTimeMPC_OSQP",
    "MLP",
    "QuadraticCost",
    "Trigonometric",
    "ReferenceSubdiagram",
]