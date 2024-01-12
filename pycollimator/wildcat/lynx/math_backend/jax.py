from jax import lax
import jax.numpy as jnp

__all__ = ["jax_functions", "jax_constants"]


def dtype(x):
    return jnp.asarray(x).dtype


jax_functions = {
    "dtype": dtype,
    "asarray": jnp.asarray,
    "array": jnp.array,
    "zeros": jnp.zeros,
    "zeros_like": jnp.zeros_like,
    "reshape": jnp.reshape,
    "cond": lax.cond,
    "sin": jnp.sin,
}

jax_constants = {
    "inf": jnp.inf,
    "nan": jnp.nan,
    "float64": jnp.float64,
    "float32": jnp.float32,
    "float16": jnp.float16,
    "int64": jnp.int64,
    "int32": jnp.int32,
    "int16": jnp.int16,
    "bool": jnp.bool_,
}
