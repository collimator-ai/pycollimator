"""
DataSource which meets the specification from design.
"""
import numpy as np
import jax.numpy as jnp
import pandas as pd
from jax import lax

from .generic import SourceBlock

__all__ = [
    "DataSource",
]


class DataSource(SourceBlock):
    def __init__(
        self,
        file_name: str,
        data_columns: str = "1",  # slice, e.g. 3:4
        extrapolation: str = "hold",
        header_as_first_row: bool = False,
        interpolation: str = "zero_order_hold",
        sampling_interval: float = 1.0,
        time_column: str = "0",  # @am. could be an int
        time_samples_as_column: bool = False,
        **kwargs,
    ):
        # FIXME: legacy compat
        if "data_integration_id" in kwargs:
            del kwargs["data_integration_id"]

        # print(f"DataSource.init(). name={name} fnam={file_name}")

        # Load and preprocess the csv
        if header_as_first_row:
            header = 0
        else:
            header = None
        # print(f"header={header}")
        usecols = self.str2colindices(data_columns)
        # print(f"usecols={usecols}")

        df = pd.read_csv(
            file_name, header=header, skipinitialspace=True, dtype=np.float64
        )

        # print(f"df after read_csv\n{df}")

        if time_samples_as_column:
            index = int(time_column)
            index = df.columns[index]
        else:
            index = self.make_time_col(0.0, sampling_interval, df.shape[0])
        # print(f"index={index}")
        # print(f"df.columns={df.columns}")
        # print(f"df.columns.isin(usecols)={df.columns.isin(usecols)}")

        # for all colmuns that are not reuested by user for output,
        # create a filter that can be used to remove these from the dataframe.
        # this ensures we get the correct output dimension, with the correct values.
        # Note, if one of the columns is time samples, we compute this filter before
        # setting the time column as the index of the dataframe because the user
        # may have provided data_columns as indices, and these would be relative to
        # the file including the time column.
        # first check if usecols and df.columns are the same identifiers
        col_filter = df.columns.isin(usecols)
        if not any(col_filter):
            # if usecols is indices, and df.coliumn is names
            # we need to create a filter that first converts usecols
            # to a list of column names
            # but first ensure that usecols range doens't go beyond
            # the number of columns
            # print(f"len(df.columns)={len(df.columns)}")
            if len(usecols) == 1:
                if usecols[0] > len(df.columns) - 1:
                    usecols[0] = len(df.columns) - 1
            else:
                lwr = min(usecols)
                # add 1 here because max(range(1,4))=3, and range(1,3) wuld be incorrect.
                upr = max(usecols) + 1
                if upr > len(df.columns) + 1:
                    upr = len(df.columns) + 1
                usecols = range(lwr, upr)

            # print(f"usecols={usecols}")
            col_filter = df.columns.isin(df.columns[usecols])
        # print(f"col_filter={col_filter} before removing time col if its in there.")

        # now that we have computed the filter based on orignla col indices of the file
        # we can move the time col to the dataframe index if that is what was requested
        df.set_index(index, inplace=True)
        # print(f"df after set index\n{df}")

        if time_samples_as_column:
            # if the time col was in the file, and it was used as the dataframe index
            # it would have been part of the filter, but we have since removed the column,
            # this means we need to remove that element of the filter.
            # del col_filter[int(time_column)]
            col_filter = np.delete(col_filter, int(time_column))
            # print(f"col_filter={col_filter} after removing time")

        df = df.loc[:, col_filter]

        # print(f"df when done\n{df}")

        times = jnp.array(df.index.to_numpy())
        data = jnp.array(df.to_numpy())
        max_i_zoh = len(times) - 1
        max_i_interp = len(times) - 2
        output_dim = data.shape[1]

        def get_below_row_idx(time, max_i):
            """
            first the clip the value of 'time' so that it falls inside the
            range of 'times'. this ensures we dont get strange extrapolation behavior.
            then, find the index of 'times' how value is largest but still smaller than
            'time'. we use this to locate the rows in 'times' that bound 'time'.
            """
            time_clipped = jnp.clip(time, times[0], times[-1])
            i = jnp.argmin(time_clipped >= times) - 1
            return i, time_clipped

        def _func_zoh(time):
            i, time_clipped = get_below_row_idx(time, max_i_zoh)
            return lax.cond(
                extrapolation == "zero" and time > times[-1],
                lambda time: jnp.zeros(output_dim),
                lambda time: data[i, :],
                time,
            )

        def _func_interp(time):
            """
            the second lambda function does this:
            y = (yp2-yp1)/(xp2-xp1)*(x-xp1) + yp1
            but does so by operating on the arrays
            ap1 and ap2 which provide the yp1's and yp2's.
            the xp1's and xp2's are time values.
            """
            i, time_clipped = get_below_row_idx(time, max_i_interp)
            ap1 = data[i, :]
            ap2 = data[i + 1, :]

            return lax.cond(
                extrapolation == "zero" and time > times[-1],
                lambda time: jnp.zeros(output_dim),
                lambda time: (ap2 - ap1)
                / (times[i + 1] - times[i])
                * (time_clipped - times[i])
                + ap1,
                time,
            )

        if interpolation == "zero_order_hold":
            _func = _func_zoh
        else:
            _func = _func_interp

        super().__init__(_func, **kwargs)

        if data.size == 0:
            raise ValueError(
                f"DataSource {self.name} did not succeed in getting the requested data columns."
            )

        self.declare_configuration_parameters(
            file_name=file_name,
            data_columns=data_columns,
            extrapolation=extrapolation,
            header_as_first_row=header_as_first_row,
            interpolation=interpolation,
            sampling_interval=sampling_interval,
            time_column=time_column,
            time_samples_as_column=time_samples_as_column,
        )

    def make_time_col(self, start, step, n):
        """
        this function computes the time vector from sample
        interval when this is requested by user
        """
        return np.array([i * step + start for i in range(n)])

    def str2colindices(self, s):
        """
        this function parses the data_columns arg and returns
        a range, list, as appropriate to filter for the requested
        columns.
        """
        slice_thing = s.split(":")
        list_thing = s.split(",")
        if s.isdigit():
            # user provided a single index as integer
            index = int(s)
            return [index]
        elif len(slice_thing) == 2:
            start = slice_thing[0]
            end = slice_thing[1]
            if start.isdigit() and end.isdigit():
                # user provided a slice, e.g. 2:4
                return range(int(start), int(end))
            else:
                raise ValueError(f"DataSource data_columns={s} is not a proper slice")
        elif len(list_thing) > 1:
            is_list_of_digits = [i.isdigit() for i in list_thing]
            if all(is_list_of_digits):
                # user prvied a list of indices as integers
                return [int(i) for i in list_thing]
            elif any(is_list_of_digits):
                raise ValueError(
                    f"DataSource data_columns={s} is not a list of either ints or col names. mixting not allowed."
                )
            else:
                # user maybe provided a list of column names
                # we cant know if any or none of the provided names
                # match until we load the csv
                return list_thing

        elif len(slice_thing) == 1 and len(list_thing) == 1:
            # might be a single col name, e.g. 'c1'
            # again, we dont know if it matches the column names
            # until we load the csv
            return [s]
        else:
            raise ValueError(f"DataSource data_columns={s} is not comprehensible")
