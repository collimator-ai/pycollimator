import jax.numpy as jnp

from ...framework import LeafSystem


class BouncingBall(LeafSystem):
    def __init__(
        self, *args, g=9.81, e=1.0, b=0.0, h0=0.0, hdot=0.0, name="ball", **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self.declare_continuous_state(2, ode=self.ode)  # Two state variables.
        self.declare_continuous_state_output(name=f"{name}:y")
        self.declare_parameter("g", g)
        self.declare_parameter("e", e)  # Resitiution coefficent (0.0 <= e <= 1.0)
        self.declare_parameter("b", b)  # Quadratic drag coefficient
        self.declare_parameter("hdot", hdot)  # Speed of floor
        self.declare_parameter("h0", h0)  # Initial floor height

        self.declare_zero_crossing(
            guard=self._signed_distance,
            reset_map=self._reset,
            name="time_reset",
            direction="positive_then_non_positive",
        )

    def ode(self, time, state, **parameters):
        g = parameters["g"]
        b = parameters["b"]
        x, v = state.continuous_state
        return jnp.array([v, -g - b * v**2 * jnp.sign(v)])

    def floor_height(self, time, state, **parameters):
        h0 = parameters["h0"]
        hdot = parameters["hdot"]
        return h0 + hdot * time

    def _signed_distance(self, time, state, **parameters):
        x, v = state.continuous_state
        h = self.floor_height(time, state, **parameters)
        return x - h

    def _reset(self, time, state, **parameters):
        # Update velocity using Newtonian restitution model.
        x, v = state.continuous_state
        e = parameters["e"]
        hdot = parameters["hdot"]
        h = self.floor_height(time, state, **parameters)

        xc_post = jnp.array([h + jnp.abs(x - h), -e * v + hdot])
        return state.with_continuous_state(xc_post)
