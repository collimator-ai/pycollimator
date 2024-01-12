import jax.numpy as jnp
from ...framework import LeafSystem


class LotkaVolterra(LeafSystem):
    def __init__(self, x0=[10.0, 10.0], alpha=1.1, beta=0.4, gamma=0.4, delta=0.1):
        super().__init__(self)
        self.declare_parameter("alpha", alpha)
        self.declare_parameter("beta", beta)
        self.declare_parameter("gamma", gamma)
        self.declare_parameter("delta", delta)
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        x, y = state.continuous_state
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        gamma = parameters["gamma"]
        delta = parameters["delta"]
        return jnp.array([(alpha * x - beta * x * y), (delta * x * y - gamma * y)])
