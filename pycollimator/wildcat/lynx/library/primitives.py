from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from equinox import tree_at
import re
import warnings
from jax import lax
from typing import TYPE_CHECKING

from ..logging import logger
from ..framework.system_base import UpstreamEvalError
from ..framework import LeafSystem, ShapeMismatchError, DtypeMismatchError
from .generic import SourceBlock, FeedthroughBlock, ReduceBlock

if TYPE_CHECKING:
    from ..framework import ContextBase


__all__ = [
    "Abs",
    "Constant",
    "Sine",
    "Clock",
    "Comparator",
    "CrossProduct",
    "DeadZone",
    "DiscreteClock",
    "DotProduct",
    "EdgeDetection",
    "Exponent",
    "Gain",
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
    "Product",
    "ProductOfElements",
    "Power",
    "IOPort",
    "Log",
    "LookupTable1d",
    "LookupTable2d",
    "Chirp",
    "Quantizer",
    "ScalarBroadcast",
    "SumOfElements",
    "Slice",
    "Stack",
    "Step",
    "SquareRoot",
    "Ramp",
    "Saturate",
    "Trigonometric",
    "DiscretePID",
    "ZeroOrderHold",
    "UnitDelay",
    "DerivativeDiscrete",
    "Integrator",
    "IntegratorDiscrete",
]


# Utilities to establish whether the state type of a block matches that
# of a specified input port.
def check_continuous_state_type(
    sys: LeafSystem, context: ContextBase, port_idx=0
) -> None:
    inp_data = sys.eval_input(context, port_idx)
    local_context = context[sys.system_id]
    xc = local_context.continuous_state
    if inp_data.shape != xc.shape:
        logger.debug(
            f"System {sys.name} shape mismatch, {inp_data.shape} != {xc.shape}"
        )
        raise ShapeMismatchError(sys.system_id)
    if inp_data.dtype != xc.dtype:
        logger.debug(
            f"System {sys.name} dtype mismatch, {inp_data.dtype} != {xc.dtype}"
        )
        raise DtypeMismatchError(sys.system_id)


# Consider combining this function and predecessor.
# Note that the discrete_state is necessarily a list.
def check_discrete_state_type(
    sys: LeafSystem, context: ContextBase, port_idx=0, state_idx=0
) -> None:
    inp_data = sys.eval_input(context, port_idx)
    local_context = context[sys.system_id]
    xd = local_context.discrete_state[state_idx]
    if inp_data.shape != xd.shape:
        logger.debug(
            f"System {sys.name} shape mismatch, {inp_data.shape} != {xd.shape}"
        )
        raise ShapeMismatchError(sys.system_id)
    if inp_data.dtype != xd.dtype:
        logger.debug(
            f"System {sys.name} dtype mismatch, {inp_data.dtype} != {xd.dtype}"
        )
        raise DtypeMismatchError(sys.system_id)


class Abs(FeedthroughBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: jnp.abs(x), *args, **kwargs)

        # Add a zero-crossing event so ODE solvers can't try to integrate
        # through a discontinuity.  (Technically the discontinuity is in
        # the derivative here, but this is how SL does it).
        self.declare_zero_crossing(self._zero_crossing, direction="crosses_zero")

    def _zero_crossing(self, time, state, u):
        return u


class MatrixInversion(FeedthroughBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: jnp.linalg.inv(x), *args, **kwargs)


class MatrixTransposition(FeedthroughBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: jnp.array(x).T, *args, **kwargs)


class SumOfElements(FeedthroughBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: jnp.sum(x), *args, **kwargs)


class ProductOfElements(FeedthroughBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: jnp.prod(x), *args, **kwargs)


class Constant(SourceBlock):
    def __init__(self, value, *args, **kwargs):
        super().__init__(self.func, *args, **kwargs)
        self.declare_parameter("value", value)

    def func(self, time, **parameters):
        return parameters["value"]


class Sine(SourceBlock):
    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0, bias=0.0, **kwargs):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.bias = bias
        super().__init__(self._eval, **kwargs)

    def _eval(self, t):
        return self.amplitude * jnp.sin(self.frequency * t + self.phase) + self.bias


class Clock(SourceBlock):
    def __init__(self, **kwargs):
        super().__init__(lambda t: jnp.array(t), **kwargs)


class Comparator(LeafSystem):
    def __init__(self, atol=1e-5, rtol=1e-8, operator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # validate operator parameter
        if operator not in ["<", "<=", ">", ">=", "==", "!="]:
            name = kwargs["name"]
            raise ValueError(
                f"Comparator block {name} has invalid selection {operator} for 'operator'"
            )
        self.declare_parameter("atol", atol)
        self.declare_parameter("rtol", rtol)
        self.declare_configuration_parameters(operator=operator)

        self.declare_input_port()
        self.declare_input_port()

        evt_direction = self._process_operator(operator, rtol, atol)

        def _compare(inputs):
            x = inputs[0]
            y = inputs[1]
            # @le need to check that jax throws exception if x or y are complex
            if operator == ">":
                return x > y
            elif operator == ">=":
                return x >= y
            elif operator == "<":
                return x < y
            elif operator == "<=":
                return x <= y
            # FIXME: this code errors out for the typical case of 2 scalar inputs
            elif operator == "==":
                # wip .. likely insufficient.
                if jnp.issubdtype(x, jnp.floating):
                    return jnp.isclose(x, y, rtol, atol)
                else:
                    return x == y
            elif operator == "!=":
                if jnp.issubdtype(x, jnp.floating):
                    return not jnp.isclose(x, y, rtol, atol)
                else:
                    return x != y
            else:
                return x > y

        def _compute_output(context):
            inputs = self.collect_inputs(context)
            return _compare(inputs)

        self.declare_output_port(_compute_output)

        self.declare_zero_crossing(self._zero_crossing, direction=evt_direction)

    def _zero_crossing(self, time, state, *inputs, **params):
        return inputs[0] - inputs[1]

    def _process_operator(self, operator, rtol, atol):
        if operator in ["<", "<="]:
            return "positive_then_non_positive"
        elif operator in [">", ">="]:
            return "negative_then_non_negative"
        else:
            return "crosses_zero"


class EdgeDetection(LeafSystem):
    def __init__(self, dt, edge_detection, initial_state, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_configuration_parameters(edge_detection=edge_detection)
        self.declare_parameter("initial_state", initial_state)

        self.declare_input_port()

        # Declare an internal variable for the previous input value
        self.prev_input_index = self.declare_discrete_state(default_value=initial_state)
        self.declare_periodic_discrete_update(
            self._update_prev_input,
            period=dt,
            offset=0.0,
            state_index=self.prev_input_index,
        )

        # Declare the output calculation
        self.output_index = self.declare_discrete_state(default_value=False)
        if edge_detection == "rising":
            _func = self._detect_rising
        elif edge_detection == "falling":
            _func = self._detect_falling
        elif edge_detection == "either":
            _func = self._detect_either
        else:
            name = kwargs["name"]
            raise ValueError(
                f"EdgeDetection block {name} has invalid selection {edge_detection} for 'edge_detection'"
            )

        self.declare_periodic_discrete_update(
            _func,
            period=dt,
            offset=0.0,
            state_index=self.output_index,
        )
        self.declare_discrete_state_output(state_index=self.output_index)

    def _update_prev_input(self, time, state, e, **params):
        # Update the e[k-1] term -> state[self.prev_input_index]
        return e

    def _detect_rising(self, time, state, e, **params):
        e_prev = state.discrete_state[self.prev_input_index]
        e_prev = jnp.array(e_prev)
        e = jnp.array(e)
        not_e_prev = jnp.logical_not(e_prev)
        return jnp.logical_and(not_e_prev, e)

    def _detect_falling(self, time, state, e, **params):
        e_prev = state.discrete_state[self.prev_input_index]
        e_prev = jnp.array(e_prev)
        e = jnp.array(e)
        not_e = jnp.logical_not(e)
        return jnp.logical_and(e_prev, not_e)

    def _detect_either(self, time, state, e, **params):
        e_prev = state.discrete_state[self.prev_input_index]
        e_prev = jnp.array(e_prev)
        e = jnp.array(e)
        not_e_prev = jnp.logical_not(e_prev)
        not_e = jnp.logical_not(e)
        rising = jnp.logical_and(not_e_prev, e)
        falling = jnp.logical_and(e_prev, not_e)
        return jnp.logical_or(rising, falling)


class DotProduct(LeafSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_input_port()
        self.declare_input_port()

        def _compute_output(context):
            inputs = self.collect_inputs(context)
            return jnp.dot(inputs[0], inputs[1])

        self.declare_output_port(_compute_output)


class CrossProduct(LeafSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_input_port()
        self.declare_input_port()

        def _compute_output(context):
            inputs = self.collect_inputs(context)
            return jnp.cross(inputs[0], inputs[1])

        self.declare_output_port(_compute_output)


class IfThenElse(LeafSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_input_port()
        self.declare_input_port()
        self.declare_input_port()

        def _compute_output(context):
            inputs = self.collect_inputs(context)
            return lax.cond(
                inputs[0],
                lambda: inputs[1],
                lambda: inputs[2],
            )

        self.declare_output_port(_compute_output)
        self.declare_zero_crossing(self._edge_detection, direction="crosses_zero")

    def _edge_detection(self, time, state, *inputs, **params):
        return lax.cond(inputs[0], lambda: 1.0, lambda: -1.0)


class DeadZone(LeafSystem):
    def __init__(self, half_range=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if half_range <= 0:
            name = kwargs["name"]
            raise ValueError(
                f"DeadZone block {name} has invalid half_range {half_range}. Must be > 0."
            )

        self.primary_input_index = self.declare_input_port()

        def _dz(inputs, half_range):
            # in_dz = -half_range < inputs[0] and inputs[0] < half_range
            return lax.cond(
                abs(inputs[0]) < half_range,
                lambda: inputs[0] * 0,
                lambda: inputs[0],
            )

        def _compute_output(context):
            inputs = self.collect_inputs(context)
            parameters = context[self.system_id].parameters
            return _dz(inputs, parameters["half_range"])

        self.declare_output_port(_compute_output)
        self.declare_parameter("half_range", half_range)
        self.declare_zero_crossing(
            self._lower_limit_event_value, direction="positive_then_non_positive"
        )
        self.declare_zero_crossing(
            self._upper_limit_event_value, direction="negative_then_non_negative"
        )

    def _lower_limit_event_value(self, time, state, *inputs, **params):
        u = inputs[self.primary_input_index]
        return u + params["half_range"]

    def _upper_limit_event_value(self, time, state, *inputs, **params):
        u = inputs[self.primary_input_index]
        return u - params["half_range"]


class DiscreteClock(LeafSystem):
    """Source block that produces the time sampled at a fixed rate.

    The block maintains the most recently sampled time as a discrete state, provided
    to the output port during the following interval.

    As a consequence of the timing rules for discrete systems, this can lead to
    some possibly surprising results if the output port signal is recorded during
    simulation.  The system is sampled at the _end_ of the interval, recording the
    value of the discrete state that was maintained _during_ the interval.  Hence,
    at time `t_[n]`, the value of the output port is recorded as `x⁻[n] = t[n-1]`,
    before updating to `x⁺[n] = t[n]`.  Graphically, a discrete clock sampled at
    100 Hz would have the following time series:

          x(t)                  ●━
            |                   ┆
        .03 |              ●━━━━○
            |              ┆
        .02 |         ●━━━━○
            |         ┆
        .01 |    ●━━━━○
            |    ┆
          0 ●━━━━○----+----+----+-- t
            0   .01  .02  .03  .04

    The recorded states are the open circles, which should be interpreted as the value
    seen by all other blocks on the interval `(t[n-1], t[n])`. The results of recording
    the output with `recorded_signals={"x": clock.output_ports[0]}` would be:

    |    n    |  t[n] |  x⁻[n] |
    |---------|-------|--------|
    |    0    |  0.01 |  0.00  |
    |    1    |  0.02 |  0.01  |
    |    2    |  0.03 |  0.02  |
    """

    def __init__(self, dt, start_time=0, name=None, system_id=None):
        super().__init__(name=name, system_id=system_id)
        dtype = dt.dtype if isinstance(dt, jnp.ndarray) else jnp.asarray(dt).dtype
        self.dt = dt

        start_time = jnp.asarray(start_time, dtype=dtype)
        offset = jnp.asarray(0.0, dtype=dtype)

        self.state_index = self.declare_discrete_state(
            default_value=start_time, dtype=dtype
        )
        self.declare_discrete_state_output(state_index=self.state_index)
        self.declare_periodic_discrete_update(
            self._update,
            period=dt,
            offset=offset,
            state_index=self.state_index,
        )

    def _update(self, time, state, **params):
        return time


class Gain(FeedthroughBlock):
    def __init__(self, gain, *args, **kwargs):
        super().__init__(lambda x, gain: gain * x, *args, **kwargs)
        self.declare_parameter("gain", gain)


class Offset(FeedthroughBlock):
    def __init__(self, offset, *args, **kwargs):
        super().__init__(lambda x, offset: x + offset, *args, **kwargs)
        self.declare_parameter("offset", offset)


class Reciprocal(FeedthroughBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: 1 / x, *args, **kwargs)


class SquareRoot(FeedthroughBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: jnp.sqrt(x), *args, **kwargs)


class ScalarBroadcast(FeedthroughBlock):
    def __init__(self, m, n, **kwargs):
        if m is not None:
            m = int(m)
        else:
            m = 0
        if n is not None:
            n = int(n)
        else:
            n = 0

        if m > 0 and n > 0:
            ones_ = jnp.ones((m, n))
        elif m > 0:
            ones_ = jnp.ones((m, 1))
        elif n > 0:
            ones_ = jnp.ones((1, n))
        else:
            raise ValueError("ScalarBroadcast at least m or n must not be None or Zero")
        super().__init__(lambda x: ones_ * x, **kwargs)
        self.declare_configuration_parameters(m=m, n=n)


class Slice(FeedthroughBlock):
    def __init__(self, slice_, *args, **kwargs):
        # validate slice_str
        # FIXME: the pattern below only works for slices up to 3 dimensions.
        pattern = re.compile(r"^\[[^[\]]*(:[^[\]]+)?(,:[^[\]]+)?(,:[^[\]]+)?\]$")
        if not pattern.match(slice_):
            name = kwargs["name"]
            raise ValueError(
                f"Slice block {name} detected invalid slice operator {slice_}"
            )

        def _func(inp):
            return eval(f"inp{slice_}")

        super().__init__(_func, *args, **kwargs)
        self.declare_configuration_parameters(slice_=slice_)


class Demultiplexer(LeafSystem):
    def __init__(self, n_out, **kwargs):
        super().__init__(**kwargs)

        self.declare_input_port()

        # Need a helper function so that the lambda captures the correct value of i
        # and doesn't use something that ends up fixed in scope.
        def _declare_output(i):
            def _callback(context):
                input_vec = self.eval_input(context)
                return input_vec[i]

            self.declare_output_port(_callback)

        for i in jnp.arange(n_out):
            _declare_output(i)


class Multiplexer(ReduceBlock):
    def __init__(self, n_in, *args, **kwargs):
        super().__init__(n_in, jnp.hstack, *args, **kwargs)


class Stack(ReduceBlock):
    def __init__(self, n_in, axis, *args, **kwargs):
        # FIXME: prsently ignoring the axis arg.
        super().__init__(n_in, jnp.stack, *args, **kwargs)
        self.declare_configuration_parameters(axis=axis)


class MatrixMultiplication(ReduceBlock):
    def __init__(self, n_in, type=None, *args, **kwargs):
        if n_in != 2:
            name = kwargs["name"]
            raise ValueError(
                f"MatrixMultiplication block {name} has invalid number of inputs {n_in}."
            )

        def _func(inputs):
            return jnp.matmul(inputs[0], inputs[1])

        super().__init__(n_in, _func, *args, **kwargs)


class Adder(ReduceBlock):
    def __init__(self, n_in, *args, operators=None, **kwargs):
        if operators is None:
            _func = sum

        else:
            signs = [1 if op == "+" else -1 for op in operators]

            def _func(inputs):
                signed_inputs = [s * u for (s, u) in zip(signs, inputs)]
                return sum(signed_inputs)

        super().__init__(n_in, _func, *args, **kwargs)
        self.declare_configuration_parameters(operators=operators)


class Product(ReduceBlock):
    def __init__(
        self,
        n_in,
        *args,
        operators=None,
        denominator_limit=None,
        divide_by_zero_behavior=None,
        **kwargs,
    ):
        # Design note (@am): the implementation below may seem overly
        # complex and convoluted, but honestly i tried so many things
        # and this is the only thing i could get to work
        if "/" in operators:
            operators = operators  # Expect "**/*", etc
            num_indices = jnp.array(
                [idx for idx, op in enumerate(operators) if op == "*"]
            )
            den_indices = jnp.array(
                [idx for idx, op in enumerate(operators) if op == "/"]
            )

            def _func(inputs):
                ain = jnp.array(inputs)
                num = jnp.take(ain, num_indices)
                den = jnp.take(ain, den_indices)
                return jnp.prod(num) / jnp.prod(den)

        else:

            def _func(inputs):
                logger.debug(f"Product inputs for {self.name}: {inputs}")
                return jnp.prod(jnp.array(inputs))

        super().__init__(n_in, _func, *args, **kwargs)
        self.declare_configuration_parameters(
            operators=operators,
            denominator_limit=denominator_limit,
            divide_by_zero_behavior=divide_by_zero_behavior,
        )


class MinMax(ReduceBlock):
    def __init__(self, n_in, operator, *args, **kwargs):
        if operator == "max":

            def _func(inputs):
                return jnp.max(jnp.array(inputs))

        elif operator == "min":

            def _func(inputs):
                return jnp.min(jnp.array(inputs))

        else:
            name = kwargs["name"]
            raise ValueError(
                f"MinMax block {name} has invalid selection {operator} for 'operator'"
            )

        self.operator = operator

        super().__init__(n_in, _func, *args, **kwargs)
        self.declare_configuration_parameters(operator=operator)
        self.declare_zero_crossing(self._zero_crossing, direction="edge_detection")

    def _zero_crossing(self, time, state, *inputs, **params):
        if self.operator == "max":

            def _evt_func(inputs):
                return jnp.argmax(jnp.array(inputs))

        else:

            def _evt_func(inputs):
                return jnp.argmin(jnp.array(inputs))

        actv_idx = _evt_func(inputs)
        # actv_idx_flt = actv_idx.dtype(jnp.float_)
        actv_idx_flt = jnp.array(actv_idx, dtype=jnp.float_)
        return actv_idx_flt


class LogicalOperator(ReduceBlock):
    def __init__(self, n_in, function, *args, **kwargs):
        if function == "or":

            def _func(inputs):
                return jnp.any(jnp.array(inputs))

        elif function == "and":

            def _func(inputs):
                return jnp.all(jnp.array(inputs))

        elif function == "not":

            def _func(inputs):
                return jnp.logical_not(jnp.array(inputs))

        elif function == "xor":
            if n_in != 2:
                name = kwargs["name"]
                raise ValueError(
                    f"LogicalOperator block {name} fucntion 'xor' only supported for 2 inputs."
                )

            # Design note (@am): xor can only be computed on 2 elements.
            # https://en.wikipedia.org/wiki/XOR_gate#More_than_two_inputs
            # that's how it can be done. I did the lazy thing first
            def _func(inputs):
                return jnp.logical_xor(jnp.array(inputs))

        elif function == "nor":

            def _func(inputs):
                return jnp.logical_not(jnp.any(jnp.array(inputs)))

        elif function == "nand":

            def _func(inputs):
                return jnp.logical_not(jnp.all(jnp.array(inputs)))

        else:
            name = kwargs["name"]
            raise ValueError(
                f"LogicalOperator block {name} has invalid selection {function} for 'function'"
            )

        self._func = _func

        super().__init__(n_in, _func, *args, **kwargs)

        self.declare_configuration_parameters(function=function)
        self.declare_zero_crossing(self._edge_detection, direction="crosses_zero")

    def _edge_detection(self, time, state, *inputs, **params):
        outp = self._func(inputs)
        return lax.cond(outp, lambda: 1.0, lambda: -1.0)


class IOPort(FeedthroughBlock):
    """Simple class for organizing input/output ports for groups/submodels.

    Since these are treated as standalone blocks in the UI rather than specific
    input/output ports exported to the parent model, it is more straightforward
    to represent them that way here as well.

    This class represents a simple one-input, one-output feedthrough block where
    the feedthrough function is an identity.  The input (resp. output) port can then
    be exported to the parent model to create an Inport (resp. Outport).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: x, *args, **kwargs)


class Log(FeedthroughBlock):
    def __init__(self, base, *args, **kwargs):
        if base == "10":
            super().__init__(lambda x: jnp.log10(x), *args, **kwargs)
        elif base == "2":
            super().__init__(lambda x: jnp.log2(x), *args, **kwargs)
        elif base == "natural":
            super().__init__(lambda x: jnp.log(x), *args, **kwargs)
        else:
            name = kwargs["name"]
            raise ValueError(
                f"Log block {name} has invalid selection {base} for 'base'"
            )
        self.declare_configuration_parameters(base=base)


class Power(FeedthroughBlock):
    def __init__(self, exponent, *args, **kwargs):
        super().__init__(lambda x, exponent: jnp.power(x, exponent), *args, **kwargs)
        self.declare_parameter("exponent", exponent)


class Exponent(FeedthroughBlock):
    def __init__(self, base, *args, **kwargs):
        if base == "exp":
            super().__init__(lambda x: jnp.exp(x), *args, **kwargs)
        elif base == "2":
            super().__init__(lambda x: jnp.exp2(x), *args, **kwargs)
        else:
            name = kwargs["name"]
            raise ValueError(
                f"Exponent block {name} has invalid selection {base} for 'base'"
            )
        self.declare_configuration_parameters(base=base)


class Quantizer(FeedthroughBlock):
    def __init__(self, interval, *args, **kwargs):
        super().__init__(
            lambda x, interval: interval * jnp.round(x / interval), *args, **kwargs
        )
        self.declare_parameter("interval", interval)


class LookupTable1d(FeedthroughBlock):
    def __init__(self, input_array, output_array, interpolation, *args, **kwargs):
        input_array_jnp = jnp.array(input_array)
        output_array_jnp = jnp.array(output_array)
        max_i = len(input_array_jnp) - 1

        def lut_linear(x, input_array, output_array):
            return jnp.interp(x, input_array_jnp, output_array_jnp)

        def lut_nearest(x, input_array, output_array):
            i = jnp.argmin(jnp.abs(input_array_jnp - x))
            i = jnp.clip(i, 0, max_i)
            return output_array_jnp[i]

        def lut_flat(x, input_array, output_array):
            i = lax.cond(
                x < input_array_jnp[1],
                lambda x: 0,
                lambda x: jnp.argmin(x >= input_array_jnp) - 1,
                x,
            )
            return output_array_jnp[i]

        if interpolation == "linear":
            super().__init__(lut_linear, *args, **kwargs)
        elif interpolation == "nearest":
            super().__init__(lut_nearest, *args, **kwargs)
        else:
            super().__init__(lut_flat, *args, **kwargs)

        self.declare_parameter("input_array", input_array)
        self.declare_parameter("output_array", output_array)
        self.declare_configuration_parameters(interpolation=interpolation)


# TODO: Finish this using interp2d
class LookupTable2d(LeafSystem):
    def __init__(
        self,
        input_x_array,
        input_y_array,
        output_table_array,
        interpolation,
        *args,
        **kwargs,
    ):
        xp = jnp.array(input_x_array)
        yp = jnp.array(input_y_array)
        zp = jnp.array(output_table_array)

        def _interp2d(inputs, fill_value=None):
            """
            Bilinear interpolation on a grid.

            Args:
                x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
                    coordinates will be clamped to lie in-bounds.
                xp, yp: 1D arrays of points specifying grid points where function values
                    are provided.
                zp: 2D array of function values. For a function `f(x, y)` this must
                    satisfy `zp[i, j] = f(xp[i], yp[j])`

            Returns:
                1D array `z` satisfying `z[i] = f(x[i], y[i])`.

            https://github.com/adam-coogan/jaxinterp2d/blob/master/src/jaxinterp2d/__init__.py
            """
            # if xp.ndim != 1 or yp.ndim != 1:
            #     raise ValueError("xp and yp must be 1D arrays")
            # if zp.shape != (xp.shape + yp.shape):
            #     raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

            x = jnp.array(inputs[0])
            y = jnp.array(inputs[1])

            ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
            iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

            # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
            z_11 = zp[ix - 1, iy - 1]
            z_21 = zp[ix, iy - 1]
            z_12 = zp[ix - 1, iy]
            z_22 = zp[ix, iy]

            z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
                xp[ix] - xp[ix - 1]
            ) * z_21
            z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
                xp[ix] - xp[ix - 1]
            ) * z_22

            z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
                yp[iy] - yp[iy - 1]
            ) * z_xy2

            if fill_value is not None:
                oob = jnp.logical_or(
                    x < xp[0],
                    jnp.logical_or(x > xp[-1], jnp.logical_or(y < yp[0], y > yp[-1])),
                )
                z = jnp.where(oob, fill_value, z)

            return z

        super().__init__(*args, **kwargs)

        self.declare_input_port()
        self.declare_input_port()

        def _compute_output(context):
            inputs = self.collect_inputs(context)
            return _interp2d(inputs)

        self.declare_output_port(_compute_output)

        self.declare_configuration_parameters(interpolation=interpolation)
        self.declare_parameter("input_x_array", input_x_array)
        self.declare_parameter("input_y_array", input_y_array)
        self.declare_parameter("output_table_array", output_table_array)


class Chirp(SourceBlock):
    def __init__(self, f0, f1, stop_time, phi=0.0, **kwargs):
        """Linear method of https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html

        There's an extra factor of 2 that doesn't seem like it's in the SciPy version.
        """

        def _func(time, stop_time, f0, f1, phi):
            f = f0 + (f1 - f0) * time / (2 * stop_time)
            return jnp.cos(f * time + phi)

        super().__init__(_func, **kwargs)
        self.declare_parameter("stop_time", stop_time)
        self.declare_parameter("f0", f0)
        self.declare_parameter("f1", f1)
        self.declare_parameter("phi", phi)


class Step(SourceBlock):
    def __init__(self, start_value=0.0, end_value=1.0, step_time=1.0, **kwargs):
        super().__init__(self.func, **kwargs)

        self.declare_parameter("step_time", step_time)
        self.declare_parameter("start_value", start_value)
        self.declare_parameter("end_value", end_value)

        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        self.declare_discrete_state(default_value=False)
        self.declare_periodic_discrete_update(
            lambda *args, **kwargs: True, period=jnp.inf, offset=step_time
        )

    def func(self, time, **parameters):
        return lax.cond(
            time >= parameters["step_time"],
            lambda: parameters["end_value"],
            lambda: parameters["start_value"],
        )


class Pulse(SourceBlock):
    def __init__(
        self, amplitude=1.0, pulse_width=0.5, period=1.0, phase_delay=0.0, **kwargs
    ):
        super().__init__(self.func, **kwargs)

        self.declare_parameter("amplitude", amplitude)
        self.declare_parameter("pulse_width", pulse_width)
        self.declare_parameter("period", period)

        if abs(phase_delay) > 1e-9:
            warnings.warn("Warning. Pulse block phase_delay not implemented.")

        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        # ad 2 events, one for the up jump, and one the down jump
        self.declare_discrete_state(default_value=False)
        self.declare_periodic_discrete_update(
            lambda *args, **kwargs: True, period=period, offset=period
        )
        self.declare_periodic_discrete_update(
            lambda *args, **kwargs: True,
            period=period,
            offset=period + period * pulse_width,
        )

    def func(self, time, **parameters):
        period_fraction = (
            jnp.remainder(time, parameters["period"]) / parameters["period"]
        )
        return lax.cond(
            period_fraction >= parameters["pulse_width"],
            lambda: 0.0,
            lambda: parameters["amplitude"],
        )


class Ramp(SourceBlock):
    def __init__(self, start_value=0.0, slope=1.0, start_time=1.0, **kwargs):
        super().__init__(self.func, **kwargs)
        self.declare_parameter("start_time", start_time)
        self.declare_parameter("start_value", start_value)
        self.declare_parameter("slope", slope)

    def func(self, time, **parameters):
        return lax.cond(
            time >= parameters["start_time"],
            lambda: parameters["slope"] * (time - parameters["start_time"])
            + parameters["start_value"],
            lambda: parameters["start_value"],
        )


class Sawtooth(SourceBlock):
    def __init__(self, amplitude=1.0, frequency=0.5, phase_delay=1.0, **kwargs):
        super().__init__(self.func, **kwargs)

        self.declare_parameter("amplitude", amplitude)
        self.declare_parameter("frequency", frequency)
        self.declare_parameter("phase_delay", phase_delay)

        # Add a dummy event so that the ODE solver doesn't try to integrate through
        # the discontinuity.
        period = 1 / frequency
        self.declare_discrete_state(default_value=False)
        self.declare_periodic_discrete_update(
            lambda *args, **kwargs: True, period=period, offset=phase_delay
        )

    def func(self, time, **parameters):
        # np.mod((t - phase_delay), (1.0 / frequency)) * amplitude
        period = 1 / parameters["frequency"]
        return (
            jnp.mod(time - parameters["phase_delay"], period) * parameters["amplitude"]
        )


class Saturate(LeafSystem):
    def __init__(
        self,
        upper_limit=None,
        enable_dynamic_upper_limit=False,
        lower_limit=None,
        enable_dynamic_lower_limit=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.primary_input_index = self.declare_input_port()
        self.enable_dynamic_upper_limit = enable_dynamic_upper_limit
        self.enable_dynamic_lower_limit = enable_dynamic_lower_limit

        if enable_dynamic_upper_limit:
            # @am. if dynamic limit, simply ignore the static limit
            self.upper_limit_index = self.declare_input_port()
        else:
            if upper_limit is None:
                upper_limit = np.inf
            self.declare_parameter("upper_limit", upper_limit)

        if enable_dynamic_lower_limit:
            # @am. if dynamic limit, simply ignore the static limit
            self.lower_limit_index = self.declare_input_port()
        else:
            if lower_limit is None:
                lower_limit = -np.inf
            self.declare_parameter("lower_limit", lower_limit)

        self.declare_output_port(self._compute_output)
        self.declare_configuration_parameters(
            enable_dynamic_upper_limit=enable_dynamic_upper_limit,
            enable_dynamic_lower_limit=enable_dynamic_lower_limit,
        )
        self.declare_zero_crossing(
            self._lower_limit_event_value, direction="positive_then_non_positive"
        )
        self.declare_zero_crossing(
            self._upper_limit_event_value, direction="negative_then_non_negative"
        )

    def _lower_limit_event_value(self, time, state, *inputs, **params):
        u = inputs[self.primary_input_index]
        if self.enable_dynamic_lower_limit:
            lim = inputs[self.lower_limit_index]
        else:
            lim = params["lower_limit"]
        return u - lim

    def _upper_limit_event_value(self, time, state, *inputs, **params):
        u = inputs[self.primary_input_index]
        if self.enable_dynamic_upper_limit:
            lim = inputs[self.upper_limit_index]
        else:
            lim = params["upper_limit"]
        return u - lim

    def _compute_output(self, context):
        inputs = self.collect_inputs(context)
        leaf_context = context[self.system_id]
        params = leaf_context.parameters

        u = inputs[self.primary_input_index]

        ulim = (
            inputs[self.upper_limit_index]
            if self.enable_dynamic_upper_limit
            else params["upper_limit"]
        )
        llim = (
            inputs[self.lower_limit_index]
            if self.enable_dynamic_lower_limit
            else params["lower_limit"]
        )

        return jnp.clip(u, llim, ulim)


class ZeroOrderHold(LeafSystem):
    """Implements a "zero-order hold" A/D conversion.

    The only tricky thing here is inferring the size and dtype of the internal state,
    since we can't know much about it ahead of time.  This is implemented here using
    `initialize_static_data`, since the root context is available and can be used to
    get the upstream inputs.  In turn, this breaks the usual pattern of using dependency
    tracking to infer feedthrough connections, so we have to manually declare that there
    is a direct feedthrough dependency here.
    """

    # @am. this block should not need state. something seems wrong here.
    # @jc. Why wouldn't it?  Every discrete block has state.  How else would it
    #   remember the previous value of the input?

    def __init__(self, dt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt

        self.declare_input_port()
        self.state_index = self.declare_discrete_state(
            default_value=None, as_array=False
        )
        # self.state_index = self.declare_discrete_state(default_value=jnp.nan)
        self.declare_periodic_discrete_update(self._update, period=dt, offset=0.0)

        self.declare_discrete_state_output()

    def _update(self, time, state, u, **params):
        # Every dt seconds, update the state to the current input value
        return u

    def initialize_static_data(self, context):
        # If building as part of a subsystem, this may not be fully connected yet.
        # That's fine, as long as it is connected by root context creation time.
        # This probably isn't a good long-term solution:
        #   see https://collimator.atlassian.net/browse/WC-51
        template_data = self._default_discrete_state[self.state_index]
        try:
            template_data = self.eval_input(context)
            self._default_discrete_state[self.state_index] = template_data
        except UpstreamEvalError:
            logger.debug(
                "ZeroOrderHold.initialize_static_data: UpstreamEvalError. "
                "Continuing without default value initialization."
            )
            pass
        return context

    def get_feedthrough(self):
        return [(0, 0)]


class UnitDelay(LeafSystem):
    def __init__(self, dt, initial_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt

        self.declare_input_port()
        self.declare_discrete_state(default_value=initial_state)
        self.declare_periodic_discrete_update(self._update, period=dt, offset=dt)
        self.declare_discrete_state_output()
        self.declare_parameter("initial_state", initial_state)

    def _update(self, time, state, u, **params):
        # u[k] -> z[k]
        return u

    def check_types(self, context):
        check_discrete_state_type(self, context, port_idx=0, state_idx=0)


class DerivativeDiscrete(LeafSystem):
    """Implements an approximation of the input signal w.r.t. time

    Uses a backwards finite-difference discretization.
    https://docs.collimator.ai/using-model-editors/block-library/discrete-blocks#derivative-discrete

    See ZeroOrderHold for notes on type inference of the input signal.
    """

    def __init__(self, dt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt

        self.declare_input_port()
        # self.declare_discrete_state(shape=shape, dtype=dtype)
        self.state_index = self.declare_discrete_state(
            default_value=None, as_array=False
        )
        self.declare_periodic_discrete_update(
            self._update_state,
            period=dt,
            offset=0.0,
            state_index=self.state_index,
        )

        self.deriv_index = self.declare_discrete_state(
            default_value=None, as_array=False
        )
        self.declare_periodic_discrete_update(
            self._update_deriv,
            period=dt,
            offset=0.0,
            state_index=self.deriv_index,
        )

        self.declare_discrete_state_output(state_index=self.deriv_index)

    def _update_state(self, time, state, u, **params):
        # Every dt seconds, update the state to the current input value
        return u

    def _update_deriv(self, time, state, u, **params):
        # Compute the finite difference derivative of the state
        x_prev = state.discrete_state[self.state_index]
        return (u - x_prev) / self.dt

    def initialize_static_data(self, context):
        """Infer the size and dtype of the internal states"""
        # If building as part of a subsystem, this may not be fully connected yet.
        # That's fine, as long as it is connected by root context creation time.
        # This probably isn't a good long-term solution:
        #   see https://collimator.atlassian.net/browse/WC-51
        try:
            template_data = self.eval_input(context)
            self._default_discrete_state[self.state_index] = template_data
            self._default_discrete_state[self.deriv_index] = template_data
        except UpstreamEvalError:
            logger.debug(
                "DerivativeDiscrete.initialize_static_data: UpstreamEvalError. "
                "Continuing without default value initialization."
            )
            pass
        return context

    def get_feedthrough(self):
        return [(0, 0)]


class DiscretePID(LeafSystem):
    # enable_trace_cache_sources = False
    # # enable_trace_time_derivatives = False
    # enable_trace_discrete_updates = False
    # enable_trace_unrestricted_updates = False

    def __init__(
        self,
        kp=1.0,
        ki=1.0,
        kd=1.0,
        dt=0.1,
        initial_state=0.0,
        *args,
        enable_external_initial_state=False,
        **kwargs,
    ):
        if enable_external_initial_state:
            raise NotImplementedError(
                "External initial state not yet implemented for DiscretePID"
            )

        super().__init__(*args, **kwargs)

        self.declare_parameter("kp", kp)
        self.declare_parameter("ki", ki)
        self.declare_parameter("kd", kd)
        self.declare_parameter("dt", dt)
        self.declare_parameter("initial_state", initial_state)

        self.declare_input_port()

        # Declare an internal state variable for the error integral term
        self.integral_index = self.declare_discrete_state(default_value=initial_state)
        self.declare_periodic_discrete_update(
            self._update_integral,
            period=dt,
            offset=0.0,
            state_index=self.integral_index,
        )

        # Declare an internal variable for the value of the previous error input value
        self.prev_input_index = self.declare_discrete_state(
            default_value=jnp.zeros_like(initial_state)
        )
        self.declare_periodic_discrete_update(
            self._update_prev_input,
            period=dt,
            offset=0.0,
            state_index=self.prev_input_index,
        )

        # Declare an internal state to track whether a valid finite difference approximation of
        #   the derivative can be computed
        self.valid_derivative_index = self.declare_discrete_state(default_value=False)
        self.declare_periodic_discrete_update(
            self._update_valid_derivative,
            period=jnp.inf,
            offset=dt,
            state_index=self.valid_derivative_index,
        )

        # Declare the output calculation of the PID control
        self.output_index = self.declare_discrete_state(
            default_value=jnp.zeros_like(initial_state)
        )
        self.declare_periodic_discrete_update(
            self._compute_output,
            period=dt,
            offset=0.0,
            state_index=self.output_index,
        )
        self.declare_discrete_state_output(state_index=self.output_index)
        # _compute_output = self.wrap_update_callback(self._compute_output)
        # self.declare_output_port(name="output", callback=_compute_output, prerequisites_of_calc=[DependencyTicket.xd])

    def _update_integral(self, time, state, e, **params):
        dt = params["dt"]
        # compute_Iterm in CMLC

        e_int = state.discrete_state[self.integral_index]
        return e_int + e * dt

    def _update_prev_input(self, time, state, e, **params):
        # Update the e[k-1] term -> state[self.prev_input_index]
        return e

    def _update_valid_derivative(self, time, state, e, **params):
        # Update the valid derivative term -> state[self.valid_derivative_index]
        #    This should be called once at an offset of `dt`, which will switch from PI to a PID control
        return True

    def _compute_output(self, time, state, e, **params):
        kp, ki, kd, dt = params["kp"], params["ki"], params["kd"], params["dt"]
        e_prev = state.discrete_state[self.prev_input_index]
        e_int = state.discrete_state[self.integral_index]

        # deriv_valid = state.discrete_state[self.valid_derivative_index]
        # return lax.cond(
        #     deriv_valid,
        #     lambda: kp*e + ki*e_int + kd*(e - e_prev)/dt,
        #     lambda: kp*e + ki*e_int,
        # )
        return kp * e + ki * e_int + kd * (e - e_prev) / dt

    def get_feedthrough(self):
        return [(0, 0)]

    def check_types(self, context):
        check_discrete_state_type(self, context, port_idx=0, state_idx=0)


# TODO:
# - Add hold with external input
# - Add limits
# - Create an IntegratorMixin to handle reset/hold/limit?


class Integrator(LeafSystem):
    def __init__(
        self,
        initial_continuous_state,
        *args,
        enable_reset=False,
        enable_limits=False,  # TODO
        lower_limit=None,  # TODO
        upper_limit=None,  # TODO
        enable_hold=False,  # TODO
        hold_trigger_method=None,  # TODO
        enable_external_reset=False,  # must match name from json
        zeno_tolerance=1e-6,
        reset_on_enter_zeno=False,
        reset_trigger_method=None,  # TODO
        dtype=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if enable_hold:
            raise NotImplementedError("Hold not yet implemented for integrator")

        if enable_limits:
            raise NotImplementedError("Limits not yet implemented for integrator")

        self.declare_configuration_parameters(
            enable_reset=enable_reset,
            enable_external_reset=enable_external_reset,
            reset_on_enter_zeno=reset_on_enter_zeno,
        )
        self.declare_parameter("initial_continuous_state", initial_continuous_state)
        self.declare_parameter("zeno_tolerance", zeno_tolerance)

        # Default initial condition unless modified in context
        self.x0 = jnp.array(initial_continuous_state, dtype=dtype)
        self.dtype = dtype if dtype is not None else self.x0.dtype

        self.xdot_index = self.declare_input_port(
            name="in_0"
        )  # One vector-valued input
        self.declare_continuous_state(default_value=self.x0, ode=self.ode)
        self.declare_continuous_state_output(name="out_0")  # One vector-valued output

        self.enable_reset = enable_reset
        self.enable_external_reset = enable_external_reset
        self.zeno_tolerance = zeno_tolerance
        self.reset_on_enter_zeno = reset_on_enter_zeno
        if enable_reset:
            # Boolean input for triggering reset
            self.reset_trigger_index = self.declare_input_port(name="reset_trigger")

            # Zeno state status
            self.zeno_index = self.declare_discrete_state(default_value=False)
            self.declare_discrete_state_output(name="zeno", state_index=self.zeno_index)

            # Number of times reset has been triggered (debugging only)
            self.counter_index = self.declare_discrete_state(default_value=0)
            self.declare_discrete_state_output(
                name="counter", state_index=self.counter_index
            )

            # Most recent reset time
            self.t_prev_reset_index = self.declare_discrete_state(
                default_value=0.0, dtype=self.dtype
            )
            self.declare_discrete_state_output(
                name="tprev", state_index=self.t_prev_reset_index
            )

            self.declare_zero_crossing(
                guard=self._guard,
                reset_map=self._reset,
                name="reset",
                direction="positive_then_non_positive",
            )

            self.declare_zero_crossing(
                guard=self._exit_zeno_guard,
                reset_map=self._exit_zeno,
                name="exit_zeno",
                direction="positive_then_non_positive",
            )

            # Optional: reset value defined by external signal
            if enable_external_reset:
                self.reset_value_index = self.declare_input_port(name="reset_value")

    def ode(self, time, state, xdot, *reset_inputs, **params):
        # Weight the ODE by a "guard" value that will be 1.0 when the reset trigger is False.
        # or 0.0 when the reset is True.  This value should only remain at 0.0 for as long as
        # the trigger is True, and should immediately return to 1.0 when the trigger is False.

        if self.enable_reset:
            in_zeno_state = state.discrete_state[self.zeno_index]
            return lax.cond(in_zeno_state, lambda: jnp.zeros_like(xdot), lambda: xdot)

        return xdot

    def _guard(self, time, state, *inputs, **params):
        # Convert boolean trigger input to a guard function value that will go from
        # 1.0 (when trigger is False) to 0.0 (when trigger is True)
        trigger = inputs[self.reset_trigger_index]
        return jnp.array(1 - trigger, dtype=self.dtype)

    def _reset(self, time, state, *inputs, **params):
        # If the distance between events is less than the tolerance, then enter the Zeno state.
        dt = time - state.discrete_state[self.t_prev_reset_index]
        state.discrete_state[self.zeno_index] = (dt - self.zeno_tolerance) <= 0
        state.discrete_state[self.t_prev_reset_index] = time

        # Handle the reset event as usual
        if self.enable_external_reset:
            xc = inputs[self.reset_value_index]
        else:
            xc = self.x0

        # For consistency with CMLC, don't reset if entering Zeno state
        new_continuous_state = lax.cond(
            state.discrete_state[self.zeno_index] & (not self.reset_on_enter_zeno),
            lambda x: x,
            lambda x: xc,
            state.continuous_state,
        )
        state = state.with_continuous_state(new_continuous_state)

        # Count number of resets (for debugging)
        state.discrete_state[self.counter_index] += 1

        logger.debug(f"Resetting to {state}")
        return state

    def _exit_zeno_guard(self, time, state, *inputs, **params):
        # This will only be active when in the Zeno state.  It monitors the boolean trigger input
        # and will go from 1.0 (when trigger=True) to 0.0 (when trigger=False)
        trigger = inputs[self.reset_trigger_index]
        return jnp.array(trigger, dtype=self.dtype)

    def _exit_zeno(self, time, state, *inputs, **params):
        state.discrete_state[self.zeno_index] = False
        return state

    def determine_active_guards(self, root_context):
        zero_crossing_events = self.zero_crossing_events.mark_all_active()

        if not self.enable_reset:
            return zero_crossing_events

        def _get_reset(events):
            return events.unrestricted_update_events[0]

        context = root_context[self.system_id]
        in_zeno_state = context.discrete_state[self.zeno_index]

        reset = lax.cond(
            in_zeno_state,
            lambda e: e.mark_inactive(),
            lambda e: e.mark_active(),
            _get_reset(zero_crossing_events),
        )

        def _get_exit_zeno(events):
            return events.unrestricted_update_events[1]

        exit_zeno = lax.cond(
            in_zeno_state,
            lambda e: e.mark_active(),
            lambda e: e.mark_inactive(),
            _get_exit_zeno(zero_crossing_events),
        )

        zero_crossing_events = tree_at(_get_reset, zero_crossing_events, reset)
        zero_crossing_events = tree_at(_get_exit_zeno, zero_crossing_events, exit_zeno)

        return zero_crossing_events

    def check_types(self, context):
        check_continuous_state_type(self, context, port_idx=0)


class IntegratorDiscrete(LeafSystem):
    def __init__(
        self,
        dt,
        initial_state,
        *args,
        enable_reset=False,
        enable_hold=False,  # TODO
        enable_limits=False,  # TODO
        lower_limit=None,  # TODO
        upper_limit=None,  # TODO
        enable_external_reset=False,  # must match name from json
        dtype=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if enable_hold:
            raise NotImplementedError(
                "Hold not yet implemented for discrete integrator"
            )

        if enable_limits:
            raise NotImplementedError(
                "Limits not yet implemented for discrete integrator"
            )

        self.declare_configuration_parameters(
            enable_reset=enable_reset,
            enable_external_reset=enable_external_reset,
        )
        self.declare_parameter("initial_state", initial_state)

        # Default initial condition unless modified in context
        self.dt = dt
        self.x0 = jnp.array(initial_state, dtype=dtype)
        self.dtype = dtype if dtype is not None else self.x0.dtype

        self.xdot_index = self.declare_input_port(
            name="in_0"
        )  # One vector-valued input
        self.declare_discrete_state(default_value=self.x0)
        self.declare_periodic_discrete_update(self._update, period=dt, offset=dt)
        self.declare_discrete_state_output(name="out_0")  # One vector-valued output

        self.enable_reset = enable_reset
        self.enable_external_reset = enable_external_reset
        if enable_reset:
            self.reset_trigger_index = self.declare_input_port(
                name="reset_trigger"
            )  # Boolean input for triggering reset

            if enable_external_reset:
                self.reset_value_index = self.declare_input_port(
                    name="reset_value"
                )  # Optional reset value

    def _update(self, time, state, *inputs, **params):
        def _reset():
            return (
                inputs[self.reset_value_index]
                if self.enable_external_reset
                else self.x0
            )

        def _integrate():
            x = state.discrete_state[0]
            xdot = inputs[self.xdot_index]
            return x + self.dt * xdot

        if self.enable_reset:
            # If the reset is high, then return the reset value
            trigger = inputs[self.reset_trigger_index]
            return lax.cond(trigger, _reset, _integrate)

        else:
            return _integrate()

    def check_types(self, context):
        check_discrete_state_type(self, context, port_idx=0, state_idx=0)


class Trigonometric(FeedthroughBlock):
    def __init__(self, function, *args, **kwargs):
        func = {
            "sin": jnp.sin,
            "cos": jnp.cos,
            "tan": jnp.tan,
            "asin": jnp.arcsin,
            "acos": jnp.arccos,
            "atan": jnp.arctan,
            "sinh": jnp.sinh,
            "cosh": jnp.cosh,
            "tanh": jnp.tanh,
            "asinh": jnp.arcsinh,
            "acosh": jnp.arccosh,
            "atanh": jnp.arctanh,
        }[function]
        super().__init__(func, *args, **kwargs)
        self.declare_configuration_parameters(function=function)
