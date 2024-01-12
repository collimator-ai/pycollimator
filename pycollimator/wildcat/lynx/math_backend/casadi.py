from functools import partial
import numpy as np
import casadi as cs

from jax import tree_util

__all__ = ["casadi_functions", "casadi_constants"]


def dtype(x):
    if isinstance(x, cs.DM):
        return np.float64
    if isinstance(x, cs.SX):
        return cs.SX
    if isinstance(x, cs.MX):
        return cs.MX


def asarray(x):
    # This will also need to handle the symbolic case
    # return cs.DM(x)
    if isinstance(x, (cs.SX, cs.MX)):
        return x
    return cs.vertcat(*x)


def array(*args, dtype=None):
    # This will also need to handle the symbolic case
    return cs.DM(*args)


def zeros(shape, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)
    if len(shape) == 0:
        return cs.DM.zeros(1, 1)
    elif len(shape) == 1:
        return cs.DM.zeros(shape[0], 1)
    return cs.DM.zeros(*shape)


def zeros_like(x):
    return cs.DM.zeros(*x.shape)


def reshape(x, shape):
    if len(shape) == 0:
        return x
    elif len(shape) == 1:
        return cs.reshape(x, shape[0], 1)
    return cs.reshape(x, *shape)


def cond(pred, true_fun, false_fun, *operands):
    return cs.if_else(
        pred,
        true_fun(*operands),
        false_fun(*operands),
    )


def sym_like(x, name="x", sym_class=cs.SX):
    if np.isscalar(x) or x.shape == ():
        return sym_class.sym(name)

    return sym_class.sym(name, *x.shape)


# TODO: Where should this live?
#  This isn't even necessary unless you want to track the names of
#  the symbolic variables. If not you can just do:
#  `tree_map(sym_like, context)`
def make_symbolic_leaf_context(context):
    sys_name = context.name

    sym_ctx = context.with_time(sym_like(context.time, name="t"))

    # Map continuous state to symbolic variables.
    sym = partial(sym_like, name=f"{sys_name}.xc")
    sym_ctx = sym_ctx.with_continuous_state(
        tree_util.tree_map(sym, context.continuous_state)
    )

    # Map discrete state to symbolic variables.
    sym = partial(sym_like, name=f"{sys_name}.xd")
    sym_ctx = sym_ctx.with_discrete_state(
        tree_util.tree_map(sym, context.discrete_state)
    )

    # Map parameters to symbolic variables.
    p = context.parameters.as_dict()
    sym_ctx.parameters._params = {
        k: sym_like(v, name=f"{sys_name}.{k}") for k, v in p.items()
    }
    return sym_ctx


casadi_functions = {
    "dtype": dtype,
    "asarray": asarray,
    "array": array,
    "zeros": zeros,
    "zeros_like": zeros_like,
    "reshape": reshape,
    "cond": cond,
    "sin": cs.sin,
}


# Typing for CasADi is limited compared to NumPy, JAX, etc.
casadi_constants = {
    "inf": cs.inf,
    "nan": float("nan"),
    "float64": np.float64,
}
