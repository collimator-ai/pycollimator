import numpy as np
import json
import os

__all__ = [
    "write_binary_results_f",
    "read_binary_results_f",
]


NP_HAS_FLOAT128 = hasattr(np, "float128")


def _map_base_type(typ):
    # @am. wish we could use python 3.10 match-case

    # 128 bit
    if NP_HAS_FLOAT128 and typ == np.float128:
        return ["Float", 128]

    # 64 bit
    elif typ == np.int64:
        return ["Int", 64]
    elif typ == np.float64:
        return ["Float", 64]

    # 32 bit
    elif typ == np.int32:
        return ["Int", 32]
    elif typ == np.float32:
        return ["Float", 32]

    # 16 bit
    elif typ == np.int16:
        return ["Int", 16]
    elif typ == np.float16:
        return ["Float", 16]

    # 8 bit
    elif typ == np.int8:
        return ["Int", 8]

    # bool
    elif typ == np.bool_:
        return "Bool"
    else:
        raise NotImplementedError(f" no mapping for type {typ}")


def _map_types(typ, shape):
    # print(f"[_map_types] shape={shape}")
    mapped_typ = _map_base_type(typ)

    if len(shape) == 0:
        return mapped_typ

    dims = [["Int", dim] for dim in shape]
    signal_type = ["Tensor", [["Dims", dims], mapped_typ]]

    return signal_type


def write_binary_results_f(results, datadir, logsdir=None):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    toc_signals = []  # for toc.json
    signal_types_signals = []  # for signal_types.json
    clock_name = "global_clock"
    clocks = [{"name": clock_name, "clock_spec": ["Continuous", "clock"]}]
    for signal_id_, signal_ in results.items():
        if signal_id_ == "time":
            # write the file
            filename = os.path.join(datadir, "clock")
            nparray = np.array(signal_)
            nparray.astype(np.float64).tofile(filename)
            continue

        # assemble the toc entry for the signal
        signal_id = signal_id_.split(".")
        # print(f"[write_binary_results] signal_={signal_}")
        # print(f"[write_binary_results] shape={np.shape(signal_)}")

        # FIXME here we double guessing dtype via cast to numpy
        file_type = np.array(signal_).dtype
        # The signal has one more dimension - the length of the stream?
        signal_type = _map_types(file_type, np.shape(signal_)[1:])

        # print(f"_map_types->{signal_type}")
        # print(f"file_type->{file_type}")

        # NOTE: this name could become too long, I'm just hoping we'll have
        # migrated far away from this format by then.
        signal_data_file = f"{signal_id_}.dat"
        signal_spec = {
            "signal_type": signal_type,
            "clock_name": clock_name,
            "signal_data_file": signal_data_file,
        }

        toc_signal = {
            "signal_id": signal_id,
            "signal_spec": signal_spec,
            "signal_name": signal_id_,
        }

        signal_types_signal = {
            "path": signal_id_,
            "port_index": 0,  # TODO/FIXME
            "cml_type": signal_type,
        }

        toc_signals.append(toc_signal)
        signal_types_signals.append(signal_types_signal)

        # write the file
        filename = os.path.join(datadir, signal_data_file)
        nparray = np.array(signal_)
        nparray.astype(file_type).tofile(filename)

    header = {"version": [0, 0, 0], "source_model": "none", "executable": ""}
    toc = {"header": header, "clocks": clocks, "signals": toc_signals}

    toc_file = os.path.join(datadir, "toc.json")
    with open(toc_file, "w") as outfile:
        json.dump(toc, outfile, indent=2, sort_keys=False)


#     if logsdir is not None:
#         signal_types_file = os.path.join(logsdir, "signal_types.json")
#         with open(signal_types_file, "w") as outfile:
#             json.dump(signal_types_signals, outfile, indent=2, sort_keys=False)


def _to_dtype(typ):
    """
    Convert a toc.json type to a dtype. Only scalars and nd-matrices supported for now.
    Presumably this would look much nicer with Python 3.10 pattern matching.
    """

    if isinstance(typ, str) and typ == "Bool":
        return np.bool_

    kind = typ[0]
    if kind == "Float":
        size = typ[1]
        if size == 16:
            return np.float16
        elif size == 32:
            return np.float32
        elif size == 64:
            return np.float64
        elif size == 128:
            return np.float128
        else:
            raise ValueError
    elif kind == "Int":
        size = typ[1]
        if size == 8:
            return np.int8
        elif size == 16:
            return np.int16
        elif size == 32:
            return np.int32
        elif size == 64:
            return np.int64
        else:
            raise ValueError
    elif kind == "Tensor":
        return _to_dtype(typ[1][1])
    # TODO: expand the options. code came from cml/python_lib/cml.py
    else:
        raise ValueError(f"Unknown type {typ}")


def read_binary_results_f(datadir):
    toc_file = os.path.join(datadir, "toc.json")
    with open(toc_file, "r") as f:
        toc_json = json.load(f)

    time = np.memmap(os.path.join(datadir, "clock"), dtype=np.float64, mode="r")
    results = {"time": time}

    for signal_ in toc_json["signals"]:
        signal_spec = signal_["signal_spec"]
        signal_id = ".".join(signal_["signal_id"])
        print(signal_spec["signal_type"])
        dtype = _to_dtype(signal_spec["signal_type"])

        results[signal_id] = np.memmap(
            os.path.join(datadir, signal_spec["signal_data_file"]),
            dtype=dtype,
            mode="r",
        )

    return results
