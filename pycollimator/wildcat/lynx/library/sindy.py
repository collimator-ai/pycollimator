"""
Wrappers for PySINDy
"""
import numpy as np
import json
import jax.numpy as jnp

import sympy as sp
import pandas as pd

import pysindy as ps
from pysindy.feature_library import GeneralizedLibrary

from lynx.framework import LeafSystem

__all__ = [
    "ContinuousTimeSindyWithControl",
]


def _read_csv(file_path):
    """Reads a CSV file and returns a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error reading {file_path}: {e}")


def _extract_columns(data, cols):
    """Extracts columns from the DataFrame based on names or indices."""
    if cols is None:
        return None

    if isinstance(cols, (str, int)):
        cols = [cols]  # convert to list if a single column name/index is provided

    if isinstance(cols, (list, tuple)):
        extracted_cols = []
        for col in cols:
            if isinstance(col, str) and col in data.columns:
                extracted_cols.append(data[col])
            elif isinstance(col, int) and col in range(data.shape[1]):
                extracted_cols.append(data.iloc[:, col])
            else:
                raise ValueError(f"Column {col} not found in data")

        return np.column_stack(extracted_cols)
    else:
        raise ValueError(
            "Columns must be specified as strings, integers, "
            "or a list/tuple of strings/integers"
        )


def _reduce(feature_names, coefficients):
    """
    Reduce the sparse SINDyc coefficients
    by removing columns with all zeros
    """
    # Identify zero columns
    zero_columns = jnp.all(coefficients == 0, axis=0)

    # Filter out zero columns from coefficients
    coefficients_filtered = coefficients[:, ~zero_columns]

    # Filter out corresponding feature names
    feature_names_filtered = [
        name for name, is_zero in zip(feature_names, zero_columns) if not is_zero
    ]

    return feature_names_filtered, coefficients_filtered


def train(
    x_train,
    u_train,
    x_dot_train,
    time,
    poly_order=None,
    fourier_n_frequencies=None,
    threshold=0.1,
):
    """Train the SINDyc model"""
    library = []

    if poly_order is not None:
        library.append(ps.PolynomialLibrary(degree=poly_order))

    if fourier_n_frequencies is not None:
        library.append(ps.FourierLibrary(n_frequencies=fourier_n_frequencies))

    optimizer = ps.STLSQ(
        threshold=threshold, alpha=0.05, max_iter=50, normalize_columns=False
    )

    model = ps.SINDy(optimizer=optimizer, feature_library=GeneralizedLibrary(library))

    model.fit(x_train, u=u_train, x_dot=x_dot_train, t=time, quiet=True)

    base_feature_names = model.feature_names
    feature_names, coefficients = _reduce(
        model.get_feature_names(), model.coefficients()
    )

    equations = model.equations()

    return equations, base_feature_names, feature_names, jnp.array(coefficients)


class ContinuousTimeSindyWithControl(LeafSystem):
    """
    Sindy with control (SINDyc) block: Continuous-time.

    This class implements System Identification (SINDy) algorithm with control inputs.

    Attributes:
        equations (list of strings): The identified system equations.
        base_feature_names (list of strings): features x_i and u_i.
        feature_names (list of strings): Composed features with basis libraries.
        coefficients (ndarray): Coefficients of the identified model.
        nx (int): Number of states in the model.
    """

    def __init__(
        self,
        file_name,
        state_columns,
        control_input_columns,
        *args,
        dt=None,
        time_column=None,
        state_derivatives_columns=None,
        threshold=0.1,
        poly_order=2,
        fourier_n_frequencies=None,
        pretrained=False,
        pretrained_file_path=None,
        coefficients=None,
        feature_names=None,
        base_feature_names=None,
        equations=None,
        initial_state=None,
        **kwargs,
    ):
        """Initializes the LeafSystem.

        Args:
            file_name (str): Path to the CSV file containing the training data.

            state_columns: Column names (str or list[str])
                          or column indices (int or list[int]) for x_train.

            control_input_columns: Column names (str or list[str])
                          or column indices (int or list[int]) for u_train.

            dt: fixed dt (float)

            time_column : Column name (str) for column index (int) for time data.
                       If time_column is provided, then fixed `dt` above will be ignored.

            state_derivatives_columns (optional): Column names (str or list[str]) or column
                                         indices (int or list[int]) for x_dot_train.

            poly_order (int): Degree of polynomial features. Set to `None` to
                                     omit this library.

            fourier_n_frequencies (int): Number of Fourier frequencies. Set to `None`
                                         to omit this library.

            pretrained (bool): If True, use a pretrained model specified by the
                               `pretrained_file_path` argument.

            pretrained_file_path (str, optional): Path to the pretrained model file.

            initial_state: Initial state of the system for propagating the ODE forward
                           during inference.
        """
        super().__init__(*args, **kwargs)

        if (pretrained is True) and (pretrained_file_path is None):
            raise ValueError(
                "Please provide `pretrained_file_path` as boolean "
                "`pretrained` is set to True"
            )

        if (pretrained is False) and (pretrained_file_path is not None):
            raise ValueError(
                "Boolean `pretrained` is False but `pretrained_file_path` "
                "is not provided."
            )

        if pretrained is False:
            if (dt is None) and (time_column is None):
                raise ValueError(
                    "Neither fixed dt nor column for time data are provided."
                )

        if (poly_order is None) and (fourier_n_frequencies is None):
            raise ValueError(
                "Please use atleast one of the Polynomial or Fourier libraries "
                "by setting `poly_order` and/or `fourier_n_frequencies` "
            )

        # validate training data from UI
        ui_pretrained_data = [coefficients, feature_names, base_feature_names]
        all_ui_pretrained_data_is_none = all(v is None for v in ui_pretrained_data)
        all_ui_pretrained_data_is_not_none = all(
            v is not None for v in ui_pretrained_data
        )

        if not (all_ui_pretrained_data_is_none or all_ui_pretrained_data_is_not_none):
            raise ValueError(
                f"Only some pretrained data from UI was provided for SINDY block: {self.name}. "
                f"{ui_pretrained_data}"
            )

        if all_ui_pretrained_data_is_not_none:
            self.base_feature_names = base_feature_names.tolist()
            self.feature_names = feature_names.tolist()
            self.coefficients = jnp.array(coefficients)

        elif pretrained:
            with open(pretrained_file_path, "r") as f:
                deserialized_model = json.load(f)

            self.equations = deserialized_model["equations"]
            self.base_feature_names = deserialized_model["base_feature_names"]
            self.feature_names = deserialized_model["feature_names"]
            self.coefficients = jnp.array(deserialized_model["coefficients"])
        else:
            df = _read_csv(file_name)

            if time_column:
                time_column_name = (
                    df.columns[time_column]
                    if isinstance(time_column, int)
                    else time_column
                )
                df = df.drop_duplicates(subset=time_column_name, keep="first")
                time = np.array(_extract_columns(df, time_column_name)[:, 0])
            else:
                time = dt

            x_train = _extract_columns(df, state_columns)
            u_train = _extract_columns(df, control_input_columns)
            x_dot_train = (
                _extract_columns(df, state_derivatives_columns)
                if state_derivatives_columns
                else None
            )

            (
                self.equations,
                self.base_feature_names,
                self.feature_names,
                self.coefficients,
            ) = train(
                x_train,
                u_train,
                x_dot_train,
                time,
                poly_order=poly_order,
                fourier_n_frequencies=fourier_n_frequencies,
                threshold=threshold,
            )

        _, self.nx = self.coefficients.shape
        self.declare_input_port()  # one vector valued input port for u

        if initial_state is None:
            initial_state = jnp.zeros(self.nx)

        self.declare_continuous_state(
            shape=(self.nx,), ode=self.ode, default_value=jnp.array(initial_state)
        )
        self.declare_continuous_state_output()  # output of the state in ODE

        # SymPy parsing to compute $\dot{x} = f(x,u)$
        sympy_base_features = sp.symbols(self.base_feature_names)

        # Convert feature names to sympy expressions
        sympy_feature_expressions = []
        for name in self.feature_names:
            # Replace spaces with multiplication
            name = name.replace(" ", "*")
            expr = sp.sympify(name)
            sympy_feature_expressions.append(expr)

        x_and_u_vec = sp.Matrix(sympy_base_features)
        self.features_func = sp.lambdify(
            (x_and_u_vec,), sympy_feature_expressions, modules="jax"
        )  # feature functions

        self.declare_continuous_state(
            shape=(self.nx,),
            ode=self.ode,
            default_value=jnp.array(initial_state),
        )

    def ode(self, time, state, inputs, **params):
        """
        The ODE system RHS. The RHS is given by
        coefficients @ features
        """
        x = state.continuous_state
        u = inputs
        x_and_u = jnp.hstack([x, u])
        features_evaluated = self.features_func(x_and_u)
        x_dot = jnp.matmul(self.coefficients, jnp.atleast_1d(features_evaluated))
        return x_dot

    def serialize(self, filename):
        """
        Save the relevant class attributes post training
        so that model state can be restored
        """
        sindy_model = {
            "equations": self.equations,
            "base_feature_names": self.base_feature_names,
            "feature_names": self.feature_names,
            "coefficients": self.coefficients.tolist(),  # Can't serialize numpy arrays
        }
        with open(filename, "w") as f:
            json.dump(sindy_model, f)
