import torch

__all__ = ["torch_functions", "torch_constants"]


def dtype(x):
    return torch.as_tensor(x).dtype


def cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    else:
        return false_fun(*operands)


def zeros_like(x):
    return torch.zeros(*x.shape, dtype=dtype(x))


torch_functions = {
    "dtype": dtype,
    "asarray": torch.as_tensor,
    "array": torch.tensor,
    "zeros": torch.zeros,
    "zeros_like": zeros_like,
    "reshape": torch.reshape,
    "cond": cond,
    "sin": torch.sin,
}

torch_constants = {
    "inf": torch.inf,
    "nan": torch.nan,
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "bool": torch.bool,
}
