import jax.numpy as jnp
from ...framework import LeafSystem


class VanDerPol(LeafSystem):
    def __init__(self, x0=[0.0, 0.0], mu=1.0):
        super().__init__(self)
        self.declare_parameter("mu", mu)
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        x, y = state.continuous_state
        mu = parameters["mu"]
        return jnp.array([y, mu * (1 - x**2) * y - x])
