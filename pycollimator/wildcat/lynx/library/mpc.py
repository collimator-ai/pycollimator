import numpy as np
import jax
from jax.scipy import linalg
import jax.numpy as jnp
import jaxopt

import osqp

# from ..logging import logger
from ..framework import LeafSystem

__all__ = [
    "LinearDiscreteTimeMPC",
    "LinearDiscreteTimeMPC_OSQP",
]


class LinearDiscreteTimeMPC(LeafSystem):
    # enable_trace_discrete_updates = False

    def __init__(
        self,
        lin_sys,
        Q,
        R,
        N,
        dt,
        x_ref,
        lbu=-np.inf,
        ubu=np.inf,
        name=None,
        warm_start=False,
    ):
        super().__init__(name=name)
        self.n = lin_sys.A.shape[0]
        self.m = lin_sys.B.shape[1]
        self.N = N

        # Convert to discrete time with Euler discretization
        A = jnp.eye(self.n) + dt * lin_sys.A
        B = dt * lin_sys.B

        self.warm_start = warm_start
        self.solver, init_params = self._make_solver(A, B, Q, R, lbu, ubu, N, x_ref)

        # Input: current state (x0)
        self.declare_input_port()

        # # State: current optimal control value (TODO: Should be a Trajectory object or similar)
        # self.state_index = self.declare_discrete_state(default_value=jnp.zeros(self.m))

        # # Output port: current optimal control value
        # self.declare_discrete_state_output(name=f"u_opt", state_index=self.state_index)

        # State: KKTSolution
        self.state_index = self.declare_discrete_state(
            default_value=init_params, as_array=False
        )

        def _output_callback(context):
            local_context = self.get_from_root(context)
            osqp_params = local_context.discrete_state[self.state_index]
            z_opt = osqp_params.primal[0].reshape((self.n + self.m, self.N), order="F")
            u_opt = z_opt[self.n :, :]
            return u_opt[:, 0]

        # Output port: current optimal control value
        self.declare_output_port(_output_callback)

        self.declare_periodic_discrete_update(
            period=dt,  # TODO: Make this possibly different from the integration timestep
            offset=0.0,
            callback=self.solver,
            state_index=self.state_index,
        )

    def _make_solver(self, A, B, Q, R, lbu, ubu, N, xf):
        from jax.experimental import sparse

        n = self.n
        m = self.m

        # Identity matrices of state and control dimension
        I_A = jnp.eye(n)
        I_B = jnp.eye(m)

        def e(k):
            """Unit vector in the kth direction"""
            return jnp.zeros(N).at[k].set(1.0)

        blocks = [Q, R] * N
        P = linalg.block_diag(*blocks)

        # The initial condition constraint is x[0] = x0
        L0 = jnp.eye(n, N * (n + m))

        # The defect constraint for step k is
        #    0 = (A * x[k] + B * u[k]) - x[k+1]
        L_defect = jnp.vstack(
            [
                jnp.kron(e(k), jnp.hstack([A, B]))
                + jnp.kron(e(k + 1), jnp.hstack([-I_A, 0 * B]))
                for k in range(N - 1)
            ]
        )

        # Constraint on terminal state
        Lf = jnp.kron(e(N - 1), jnp.hstack([I_A, 0 * B]))

        # Constraints on the control input
        L_input = jnp.vstack(
            [jnp.kron(e(k), jnp.hstack([0 * B.T, I_B])) for k in range(N)]
        )

        # Stack the constraint matrices and define bounds
        #  lb <= Lx <= ub
        L = jnp.vstack([L0, L_defect, Lf, L_input])

        def _get_bounds(x0):
            lb = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, lbu)]
            )
            ub = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, ubu)]
            )
            return lb, ub

        # self.qp = jaxopt.BoxOSQP(matvec_Q=_matvec_Q, matvec_A=_matvec_A)
        c = jnp.zeros(N * (n + m))

        # qp = jaxopt.BoxOSQP()

        P_sp = sparse.BCOO.fromdense(P)
        L_sp = sparse.BCOO.fromdense(L)

        # @sparse.sparsify
        @jax.jit
        def _matvec_Q(params_Q, x):
            """Matrix-vector product Q * x"""
            return P_sp @ x

        @jax.jit
        def _matvec_A(params_A, x):
            """Matrix-vector product A * x"""
            return L_sp @ x

        self.qp = jaxopt.BoxOSQP(matvec_Q=_matvec_Q, matvec_A=_matvec_A)

        lb, ub = _get_bounds(xf)
        z0 = jnp.zeros(N * (n + m))
        init_params = self.qp.init_params(
            z0, params_obj=(None, c), params_eq=None, params_ineq=(lb, ub)
        )

        def _solve(time, state, x0):
            lb, ub = _get_bounds(x0)

            if self.warm_start:
                init_params = state.discrete_state[self.state_index]
            else:
                init_params = None
            # sol = qp.run(params_obj=(P, c), params_eq=L, params_ineq=(lb, ub)).params
            # sol = self.qp.run(params_obj=(None, c), params_ineq=(lb, ub)).params
            return self.qp.run(
                init_params=init_params,
                params_obj=(None, c),
                params_ineq=(lb, ub),
            ).params

        return jax.jit(_solve), init_params


class LinearDiscreteTimeMPC_OSQP(LeafSystem):
    """
    Same as above, but using OSQP.  This is an example of a case where a traced array gets passed
    to a function that doesn't know how to handle it.
    """

    enable_trace_discrete_updates = False

    def __init__(
        self,
        lin_sys,
        Q,
        R,
        N,
        dt,
        x_ref,
        lbu=-np.inf,
        ubu=np.inf,
        name=None,
    ):
        self.enable_trace_discrete_updates = False
        super().__init__(name=name)
        self.n = lin_sys.A.shape[0]
        self.m = lin_sys.B.shape[1]
        self.N = N

        # Input: current state (x0)
        self.declare_input_port()

        # State: current optimal control value (TODO: Should be a Trajectory object or similar)
        self.state_index = self.declare_discrete_state(default_value=jnp.zeros(self.m))

        # Output port: current optimal control value
        self.declare_discrete_state_output(name="u_opt", state_index=self.state_index)

        # Convert to discrete time with Euler discretization
        A = jnp.eye(self.n) + dt * lin_sys.A
        B = dt * lin_sys.B

        self._make_solver(A, B, Q, R, lbu, ubu, N, x_ref)

        self.declare_periodic_discrete_update(
            period=dt,  # TODO: Make this possibly different from the integration timestep
            offset=0.0,
            callback=self.solve,
            state_index=self.state_index,
            enable_tracing=False,
        )

        assert not self._periodic_events.discrete_update_events[0].enable_tracing

    def solve(self, time, state, x0):
        lb, ub = self.get_bounds(x0)

        self.solver.update(l=np.array(lb), u=np.array(ub))

        # Solve problem
        sol = self.solver.solve()

        z_opt = np.reshape(sol.x, (self.n + self.m, self.N), order="F")

        # Split solution into states and controls
        # x_opt = z_opt[:self.n, :]
        u_opt = z_opt[self.n :, :]

        # Return current best projected action
        return u_opt[:, 0]

    def _make_solver(self, A, B, Q, R, lbu, ubu, N, xf):
        from scipy import sparse

        n = self.n
        m = self.m

        # Identity matrices of state and control dimension
        I_A = jnp.eye(n)
        I_B = jnp.eye(m)

        def e(k):
            """Unit vector in the kth direction"""
            return jnp.zeros(N).at[k].set(1.0)

        blocks = [Q, R] * N
        P = linalg.block_diag(*blocks)

        # The initial condition constraint is x[0] = x0
        L0 = jnp.eye(n, N * (n + m))

        # The defect constraint for step k is
        #    0 = (A * x[k] + B * u[k]) - x[k+1]
        L_defect = jnp.vstack(
            [
                jnp.kron(e(k), jnp.hstack([A, B]))
                + jnp.kron(e(k + 1), jnp.hstack([-I_A, 0 * B]))
                for k in range(N - 1)
            ]
        )

        # Constraint on terminal state
        Lf = jnp.kron(e(N - 1), jnp.hstack([I_A, 0 * B]))

        # Constraints on the control input
        L_input = jnp.vstack(
            [jnp.kron(e(k), jnp.hstack([0 * B.T, I_B])) for k in range(N)]
        )

        # Stack the constraint matrices and define bounds
        #  lb <= Lx <= ub
        L = jnp.vstack([L0, L_defect, Lf, L_input])

        def get_bounds(x0):
            lb = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, lbu)]
            )
            ub = jnp.hstack(
                [x0, jnp.zeros(L_defect.shape[0]), xf, jnp.full(N * m, ubu)]
            )
            return lb, ub

        self.get_bounds = jax.jit(get_bounds)
        self.solver = osqp.OSQP()

        lb, ub = get_bounds(jnp.zeros(n))  # Initialize solver with dummy variables
        self.solver.setup(
            P=sparse.csc_matrix(P),
            A=sparse.csc_matrix(L),
            l=np.array(lb),
            u=np.array(ub),
            verbose=False,
        )
