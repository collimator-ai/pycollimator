import io
import control

import numpy as np
import pandas as pd

from pycollimator.error import NotFoundError
from pycollimator.models import Model
from pycollimator.utils import is_path
from pycollimator.diagrams import Block


class SimulationResults:
    def __init__(self, stream, model: Model):
        if isinstance(stream, str):
            stream = io.StringIO(stream)
        self._raw_df = pd.read_csv(stream)
        self._model = model

    def __repr__(self) -> str:
        try:
            df = self.to_pandas()
            return f"<{self.__class__.__name__} {len(df.index)} rows x {len(df.columns)} columns>"
        except NotFoundError:
            return f"<{self.__class__.__name__} 0 rows x 0 columns>"

    def __getitem__(self, item):
        if item == "time":
            return self._raw_df["time"]  # pylint: disable=unsubscriptable-object
        if isinstance(item, Block):
            blk = item
        elif is_path(item):
            return self.to_pandas(path=item)
        else:
            # todo: do a search the model graph that constructs path 1
            try:
                blk = self._model.find_block(name=item, case=True)
            except NotFoundError:
                try:
                    blk = self._model.find_block(name=item, case=False)
                except NotFoundError:
                    blk = self._model.find_block(item)
        df = self.to_pandas(path=blk.path)
        return df

    def to_pandas(
        self,
        pattern=None,
        name=None,
        path=None,
        type=None,
        case=False,
        item=None,
    ) -> pd.DataFrame:
        # if nothing is specified, return all columns
        if (
            pattern is None
            and name is None
            and path is None
            and type is None
            and item is None
        ):
            cols = self._raw_df.columns[1:]  # remove time column and add back as index
        else:  # find specific block(s) to return results of
            # matches regardless of port specified or not, as well as all outputs of specified block
            if is_path(pattern):
                block_paths = [pattern]
            elif path:
                block_paths = [path]
            elif isinstance(item, Block):
                block_paths = [item.path]
            else:  # when given name or type, must query for blocks first.
                blocks = self._model.find_blocks(
                    pattern=pattern, name=name, type=type, case=case
                )
                block_paths = []
                for block in blocks:
                    block_paths.append(block.path)
            cols = []
            for col in self._raw_df.columns:
                last_index = col.rfind(".")
                block_path = col[: col.rfind(".")]
                if (last_index != -1 and (block_path in block_paths)) or (
                    col in block_paths
                ):
                    cols.append(col)
        if len(cols) == 0:
            raise NotFoundError(
                (
                    f"Simulation of model '{self._model}' has no results for "
                    f"(pattern='{pattern}' name='{name}' path='{path}' type='{type}' case='{case}' item='{item}') "
                )
            )

        df = self._raw_df[cols]  # pylint: disable=unsubscriptable-object
        df.columns = cols
        df.index = self._raw_df["time"]  # pylint: disable=unsubscriptable-object
        df.index.name = "time"
        return df

    @property
    def columns(self) -> list:
        return self.to_pandas().columns.to_list()


class LinearizationResult:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def to_state_space(self):
        """
        convert the linearization result to a python control object
        """
        return control.StateSpace(self.A, self.B, self.C, self.D)

    def __repr__(self):
        def tostring(x):
            return np.array2string(x, separator=",").replace("\n", "").replace(" ", "")

        return (
            f"<{self.__class__.__name__} A={tostring(self.A)} B={tostring(self.B)} "
            + f"C={tostring(self.C)} D={tostring(self.D)}>"
        )

    @classmethod
    def _from_csv(cls, csv_text):
        a_mat, b_mat, c_mat, d_mat = cls.__reshape_results_linearization(csv_text)
        return cls(a_mat, b_mat, c_mat, d_mat)

    # FIXME this format is not very robust
    @classmethod
    def __reshape_results_linearization(csv, results_text):
        read_st = -1
        lines = results_text.splitlines()
        for each in lines:
            if read_st == -1:
                if each[:3] == "dim":
                    # do the processing common for all matrices to get their dimensions
                    line_data = each.split(",")
                    mat_name = line_data[1]
                    tmp_dim = np.fromiter((float(x) for x in line_data[2:4]), dtype=int)
                    read_st = 0
                    if mat_name == "A":
                        a_dims = tmp_dim.copy()
                    elif mat_name == "B":
                        b_dims = tmp_dim.copy()
                    elif mat_name == "C":
                        c_dims = tmp_dim.copy()
                    elif mat_name == "D":
                        d_dims = tmp_dim.copy()
                    else:
                        print("unrecognized matrix name")
            else:
                # do the processing common for all matrices tobget their data
                line_data = each.split(",")
                line_data = line_data[:-1]
                # print(line_data)
                tmp = np.fromiter((float(x) for x in line_data), dtype=float)
                read_st = -1
                if mat_name == "A":
                    a_mat = np.reshape(tmp, a_dims)
                elif mat_name == "B":
                    b_mat = np.reshape(tmp, b_dims)
                elif mat_name == "C":
                    c_mat = np.reshape(tmp, c_dims)
                elif mat_name == "D":
                    d_mat = np.reshape(tmp, d_dims)
                else:
                    print("unrecognized matrix name")
        # get column names instead of uuids.
        return a_mat, b_mat, c_mat, d_mat
