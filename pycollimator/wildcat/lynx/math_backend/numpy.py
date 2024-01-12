import numpy as np

__all__ = ["numpy_functions", "numpy_constants"]


def dtype(x):
    return np.asarray(x).dtype


def cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


numpy_functions = {
    "dtype": dtype,
    "asarray": np.asarray,
    "array": np.array,
    "zeros": np.zeros,
    "zeros_like": np.zeros_like,
    "reshape": np.reshape,
    "cond": cond,
    "sin": np.sin,
}

numpy_constants = {
    "inf": np.inf,
    "nan": np.nan,
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "bool": np.bool_,
}
