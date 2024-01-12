from timeit import timeit
from functools import partial
import numpy as np
import jax

from ..simulation import odeint, simulate, ODESolverOptions, SimulatorOptions

# from ..profiling import Profiler
from ..logging import logger


def make_benchmark(
    system,
    t0,
    tf,
    rtol=1e-6,
    atol=1e-8,
    max_steps=16**3,
    max_major_steps=100,
    use_simulator=True,
    run_once=True,
    context=None,
    recorded_signals=None,
):
    if context is None:
        context = system.create_context()

    ode_options = ODESolverOptions(
        rtol=rtol,
        atol=atol,
        # save_steps=False,
        max_steps=max_steps,
        method="default",
        # method="Dopri5",
        # max_steps=100,
        # method="RK4",
    )

    sim_options = SimulatorOptions(
        max_major_steps=max_major_steps,
        return_context=False,
    )

    # @jax.jit
    def _odeint():
        sol = odeint(system, context, (t0, tf), ode_options=ode_options)
        return sol

    if use_simulator:
        _run = partial(
            simulate,
            system=system,
            ode_options=ode_options,
            options=sim_options,
            context=context,
            recorded_signals=recorded_signals,
            tspan=(t0, tf),
        )
    else:
        _run = jax.jit(_odeint)

    if run_once:
        _run()  # Run once to make sure everything is compiled

    return _run


class Benchmark:
    """Time JIT compilation and simulation separately

    Example usage for a model from the UI:

    ```
    if __name__ == "__main__":
        testdir = "."
        model_json = "model.json"
        model = lynx.load_model(testdir, model=model_json)

        system = model.diagram
        context = system.create_context()

        profiler = Profiler(system, sim_stop_time=10.0)
        profiler.time()
    ```
    """

    def __init__(
        self,
        system,
        context=None,
        sim_start_time=0.0,
        sim_stop_time=10.0,
        rtol=1e-6,
        atol=1e-8,
        max_steps=16**3,
        max_major_steps=100,
        recorded_signals=None,
    ):
        self.system = system

        if context is None:
            context = system.create_context()
        self.context = context

        self.sim_start_time = sim_start_time
        self.sim_stop_time = sim_stop_time
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.max_major_steps = max_major_steps
        self.recorded_signals = recorded_signals

    def _make_benchmark(self, jit=True, context=None):
        return make_benchmark(
            self.system,
            self.sim_start_time,
            self.sim_stop_time,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            max_major_steps=self.max_major_steps,
            run_once=jit,
            context=context,
            recorded_signals=self.recorded_signals,
        )

    def _timeit(self, run, N=1):
        _globals = globals()
        _globals["run"] = run
        return (1 / N) * timeit(
            "run()",
            number=N,
            globals=_globals,
        )

    def time_total(self, N=1):
        """Test creating context, compiling, and simulating"""

        def _run():
            return self._make_benchmark(jit=True, context=None)

        return self._timeit(_run, N=N)

    def time_context_create(self, N=1):
        """Test creating the context only"""

        def _run():
            return self._make_benchmark(jit=False, context=None)

        return self._timeit(_run, N=N)

    def time_compile_and_sim(self, N=1):
        """Test compiling and simulating only, without context creation"""

        def _run():
            return self._make_benchmark(jit=True, context=self.context)

        return self._timeit(_run, N=N)

    def time_sim(self, N=1):
        """Test simulation time only, without context creation or compilation"""
        _run = self._make_benchmark(jit=True, context=self.context)
        return self._timeit(_run, N=N)

    def time(self, N_compile=1, N_sim=1):
        compile_and_sim_time = self.time_compile_and_sim(N=N_compile)
        sim_time = self.time_sim(N=N_sim)
        compile_time = compile_and_sim_time - sim_time
        logger.info(f"{compile_time=}")
        logger.info(f"{sim_time=}")


def fd_grad(func, *inputs, eps=1e-6):
    """Compute the gradient of a function using finite differencing.

    Assume the function has a single scalar output, but possibly multiple vector-
    valued inputs.
    """

    nominal_output = func(*inputs)

    grads = [np.zeros_like(x) for x in inputs]

    perturbed_inputs = list(inputs).copy()
    for i, x0 in enumerate(inputs):
        if np.isscalar(x0):
            perturbed_inputs[i] = x0 + eps
            perturbed_output = func(*perturbed_inputs)
            grads[i] = (perturbed_output - nominal_output) / eps
        else:
            x0 = np.asarray(x0)
            for j in range(x0.size):
                perturbed_inputs[i] = x0.copy()
                perturbed_inputs[i][j] += eps
                perturbed_output = func(*perturbed_inputs)
                grads[i][j] = (perturbed_output - nominal_output) / eps
        perturbed_inputs[i] = x0

    return grads


def test_single_input():
    def f(x):
        return np.dot(x, x)

    # Scalar input
    x0 = np.array([1.0])
    assert np.allclose(fd_grad(f, x0), 2.0 * x0)

    # Vector input
    x0 = np.array([1.0, 2.0])
    assert np.allclose(fd_grad(f, x0), 2.0 * x0)


def test_multiple_inputs():
    def f(x, y):
        return np.dot(x, y)

    x0 = np.array([1.0, 2.0])
    y0 = np.array([3.0, 4.0])
    assert np.allclose(fd_grad(f, x0, y0), [y0, x0])
