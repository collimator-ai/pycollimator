import jax.numpy as jnp
from ...framework import LeafSystem, DiagramBuilder

from ...library import (
    Demultiplexer,
    Multiplexer,
    Gain,
    Integrator,
    FeedthroughBlock,
    Adder,
)


class Pendulum(LeafSystem):
    def __init__(
        self,
        *args,
        x0=[1.0, 0.0],
        g=9.81,
        L=1.0,
        b=0.0,
        input_port=False,
        full_state_output=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.declare_parameter("g", g)
        self.declare_parameter("L", L)
        self.declare_parameter("b", b)
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)

        if input_port:
            self.declare_input_port(name="u")

        if full_state_output:
            self.declare_continuous_state_output(name="x")
        else:

            def _obs_callback(context):
                return context.continuous_state[0]

            self.declare_output_port(_obs_callback, name="y")

    def ode(self, time, state, *inputs, **parameters):
        θ, dθ = state.continuous_state
        g = parameters["g"]
        L = parameters["L"]
        b = parameters["b"]
        ddθ = -(g / L) * jnp.sin(θ) - b * dθ

        if inputs:
            (tau,) = inputs
            ddθ += jnp.reshape(tau, ())

        return jnp.array([dθ, ddθ])


def PendulumDiagram_(x0=[1.0, 0.0], g=9.81, L=1.0):
    builder = DiagramBuilder()

    Integrator_0 = builder.add(Integrator(x0))
    Demux_0 = builder.add(Demultiplexer(2))
    # pylint: disable=no-member
    builder.connect(Integrator_0.output_ports[0], Demux_0.input_ports[0])

    Sine_0 = builder.add(FeedthroughBlock(lambda x: jnp.sin(x), name="Sine_0"))
    builder.connect(Demux_0.output_ports[0], Sine_0.input_ports[0])

    Gain_0 = builder.add(Gain(-g / L, name="Gain_0"))
    builder.connect(Sine_0.output_ports[0], Gain_0.input_ports[0])

    Mux_0 = builder.add(Multiplexer(2))
    builder.connect(Demux_0.output_ports[1], Mux_0.input_ports[0])
    builder.connect(Gain_0.output_ports[0], Mux_0.input_ports[1])
    builder.connect(Mux_0.output_ports[0], Integrator_0.input_ports[0])
    # pylint: enable=no-member

    return builder.build()


def PendulumDiagram(
    x0=[1.0, 0.0], g=9.81, L=1.0, b=0.0, input_port=False, full_state_output=False
):
    builder = DiagramBuilder()

    Integrator_0 = builder.add(Integrator(x0, name="Integrator_0"))
    Demux_0 = builder.add(Demultiplexer(2, name="Demux_0"))
    builder.connect(Integrator_0.output_ports[0], Demux_0.input_ports[0])

    Sine_0 = builder.add(FeedthroughBlock(lambda x: jnp.sin(x), name="Sine_0"))
    builder.connect(Demux_0.output_ports[0], Sine_0.input_ports[0])

    Gain_0 = builder.add(Gain(-g / L, name="Gain_0"))
    builder.connect(Sine_0.output_ports[0], Gain_0.input_ports[0])

    # Damping
    Gain_1 = builder.add(Gain(-b, name="Gain_1"))
    builder.connect(Demux_0.output_ports[1], Gain_1.input_ports[0])

    Adder_0 = builder.add(Adder(2, name="Adder_0"))
    builder.connect(Gain_0.output_ports[0], Adder_0.input_ports[0])
    builder.connect(Gain_1.output_ports[0], Adder_0.input_ports[1])

    # Add an optional torque input
    if input_port:
        Adder_1 = builder.add(Adder(2, name="Adder_1"))
        builder.connect(Adder_0.output_ports[0], Adder_1.input_ports[0])

        Mux_0 = builder.add(Multiplexer(2, name="Mux_0"))
        builder.connect(Demux_0.output_ports[1], Mux_0.input_ports[0])
        builder.connect(Adder_1.output_ports[0], Mux_0.input_ports[1])
        builder.connect(Mux_0.output_ports[0], Integrator_0.input_ports[0])

        # Diagram-level inport
        builder.export_input(Adder_1.input_ports[1])
    else:
        Mux_0 = builder.add(Multiplexer(2, name="Mux_0"))
        builder.connect(Demux_0.output_ports[1], Mux_0.input_ports[0])
        builder.connect(Adder_0.output_ports[0], Mux_0.input_ports[1])
        builder.connect(Mux_0.output_ports[0], Integrator_0.input_ports[0])

    if full_state_output:
        builder.export_output(Integrator_0.output_ports[0])
    else:
        # Partial observation
        Demux_1 = builder.add(Demultiplexer(2, name="Demux_1"))
        builder.connect(Integrator_0.output_ports[0], Demux_1.input_ports[0])
        builder.export_output(Demux_1.output_ports[0])

    return builder.build(name="pendulum")
