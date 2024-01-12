from functools import partial


import jax
import jax.numpy as jnp

from ..framework import LeafSystem

import control  # For formatting state-space systems

from scipy import signal


__all__ = [
    "LTISystem",
    "TransferFunction",
    "linearize",
    "PID",
    "Derivative",
]


class LTISystem(LeafSystem):
    def __init__(self, A, B, C, D, initialize_states=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.declare_parameter("A", A)
        self.declare_parameter("B", B)
        self.declare_parameter("C", C)
        self.declare_parameter("D", D)

        if initialize_states is not None:
            self.declare_parameter("initialize_states", initialize_states)

        A = jnp.array(A)
        B = jnp.array(B)
        C = jnp.array(C)
        D = jnp.array(D)

        # Size checks. UI allows vectors, but here we need matrices, so reshape as needed.
        if len(A.shape) == 1:
            n = 1
            A = A.reshape((n, n))
        else:
            n = A.shape[0]
        if len(B.shape) == 1:
            m = 1
            B = B.reshape((n, m))
        else:
            m = B.shape[1]
        if len(C.shape) == 1:
            p = 1
            C = C.reshape((p, n))
        else:
            p = C.shape[0]
        if len(D.shape) == 1:
            D = D.reshape((p, m))

        assert A.shape == (n, n)
        assert B.shape == (n, m)
        assert C.shape == (p, n)
        assert D.shape == (p, m)

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.is_feedthrough = not jnp.allclose(self.D, 0.0)
        self.scalar_output = p == 1

        self.declare_input_port()  # Single input port (u)
        if isinstance(initialize_states, float) and n > 1:
            initialize_states = jnp.ones(n) * initialize_states
        self.declare_continuous_state(
            shape=(n,),
            ode=self.ode,
            default_value=initialize_states,
        )  # Single continuous state (x)

        default_output = jnp.zeros(p) if p > 1 else 0.0
        self.declare_output_port(
            self._eval_output, default_value=default_output
        )  # Single output port (y)

    def _eval_output(self, context):
        x = context[self.system_id].continuous_state
        y = jnp.matmul(self.C, jnp.atleast_1d(x))

        if self.is_feedthrough:
            u = self.eval_input(context)
            y += jnp.matmul(self.D, jnp.atleast_1d(u))

        # Handle the special case of scalar output
        if self.scalar_output:
            y = y[0]

        return y

    def ode(self, time, state, u, **params):
        x = state.continuous_state
        Ax = jnp.matmul(self.A, jnp.atleast_1d(x))
        Bu = jnp.matmul(self.B, jnp.atleast_1d(u))
        return Ax + Bu

    def __repr__(self):
        return control.ss(self.A, self.B, self.C, self.D).__repr__()

    def __str__(self):
        return control.ss(self.A, self.B, self.C, self.D).__str__()

    def _repr_latex_(self):
        return control.ss(self.A, self.B, self.C, self.D)._repr_latex_()

    def get_feedthrough(self):
        return [(0, 0)] if self.is_feedthrough else []


class TransferFunction(LTISystem):
    def __init__(self, num, den, *args, **kwargs):
        A, B, C, D = signal.tf2ss(num, den)
        super().__init__(A, B, C, D, *args, **kwargs)
        self.declare_parameter("num", num)
        self.declare_parameter("den", den)


class PID(LTISystem):
    def __init__(
        self,
        kp,
        ki,
        kd,
        n,
        initial_state=0.0,
        *args,
        enable_external_initial_state=False,
        **kwargs,
    ):
        if enable_external_initial_state:
            raise NotImplementedError(
                "External initial state not yet implemented for PID"
            )

        A = jnp.array([[0.0, 1.0], [0.0, n]])
        B = jnp.array([[0.0], [1.0]])
        C = jnp.array([(ki * n), ((kp * n + ki) - (kp + kd * n) * n)])
        D = jnp.array([(kp + kd * n)])
        initialize_states = jnp.array([initial_state, 0.0])
        super().__init__(
            A, B, C, D, initialize_states=initialize_states, *args, **kwargs
        )
        self.declare_parameter("kp", kp)
        self.declare_parameter("ki", ki)
        self.declare_parameter("kd", kd)
        self.declare_parameter("n", n)
        self.declare_parameter("initial_state", initial_state)


class Derivative(LTISystem):
    def __init__(self, N, *args, **kwargs):
        num = [N, 0]
        den = [1, N]
        A, B, C, D = signal.tf2ss(num, den)
        super().__init__(A, B, C, D, *args, **kwargs)
        self.declare_parameter("N", N)


def linearize(system, base_context, name=None):
    """Linearize the system about an operating point specified by the base context

    For now, only implemented for systems with one each (vector-valued) input and output.
    """
    xc0 = base_context.continuous_state
    u0 = system.input_ports[0].eval(base_context)

    # Map from (state, inputs) to (state derivatives, outputs)
    @jax.jit
    def f(xc, u):
        context = base_context.with_continuous_state(xc)
        system.input_ports[0].fix_value(u)
        xdot = system.eval_time_derivatives(context)
        y = system.output_ports[0].eval(context)
        return xdot, y

    @jax.jit
    def jac(xc, u):
        primals, tangents = jax.jvp(f, (xc0, u0), (xc, u))
        return tangents

    return LTISystem(*_jvp_to_ss(jac, xc0, u0), name=name)


@partial(jax.jit, static_argnums=(0,))
def _jvp_to_ss(J, xc0, u0):
    """Create a state-space realization from a function that evaluates the jacobian-vector product of a system.

    The ordering is not unique, but is deterministically chosen by the order of the flattened state vector produced
    by `ravel_pytree`."""
    from jax.flatten_util import ravel_pytree

    vec, unflatten = ravel_pytree(xc0)
    _, y0 = J(xc0, u0)  # One forward evaluation

    n = vec.size
    p = u0.size
    m = y0.size
    A = jnp.zeros((n, n))
    B = jnp.zeros((n, p))
    C = jnp.zeros((m, n))
    D = jnp.zeros((m, p))

    ucp = jnp.zeros_like(u0)
    for i in range(n):
        xcp = jnp.zeros_like(vec).at[i].set(1.0)
        xdot, y = J(unflatten(xcp), ucp)
        xdot, _ = ravel_pytree(xdot)
        y, _ = ravel_pytree(y)
        A = A.at[:, i].set(xdot)
        C = C.at[:, i].set(y)

    xcp = jnp.zeros_like(vec)
    for i in range(p):
        ucp = jnp.zeros_like(u0).at[i].set(1.0)
        xdot, y = J(unflatten(xcp), ucp)
        xdot, _ = ravel_pytree(xdot)
        y, _ = ravel_pytree(y)
        B = B.at[:, i].set(xdot)
        D = D.at[:, i].set(y)

    return (A, B, C, D)
