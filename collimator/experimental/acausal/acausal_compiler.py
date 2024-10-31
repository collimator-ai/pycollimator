# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

from typing import TYPE_CHECKING
import numpy as np
from collimator.backend import numpy_api as cnp
from collimator.experimental.acausal.component_library.base import EqnEnv, SymKind
from collimator.experimental.acausal.diagram_processing import DiagramProcessing
from collimator.experimental.acausal.acausal_diagram import AcausalDiagram
from collimator.experimental.acausal.index_reduction.index_reduction import (
    IndexReduction,
    SemiExplicitDAE,
)
from collimator.experimental.acausal.index_reduction.equation_utils import (
    compute_initial_conditions,
)
from collimator.experimental.acausal.error import (
    AcausalModelError,
    AcausalCompilerError,
)
from collimator.framework import DependencyTicket, LeafSystem, build_recorder
from collimator.framework.system_base import UpstreamEvalError
from collimator.lazy_loader import LazyLoader
from collimator.framework.system_base import Parameter

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")


if TYPE_CHECKING:
    from collimator.dashboard.serialization.from_model_json import AcausalNetwork


class AcausalSystem(LeafSystem):
    # FIXME these three are quite the hack
    acausal_network: "AcausalNetwork" = None
    outports_maps: dict[str, dict[int, int]] = None
    inports_maps: dict[str, dict[int, int]] = None

    def __init__(
        self, dp: DiagramProcessing, sed: SemiExplicitDAE, name: str, leaf_backend="jax"
    ):
        super().__init__(name=name)
        self.dp = dp
        self.sed = sed

        # Input Ports
        self._configure_inputs(dp.diagram.input_syms)

        # Parameters in Context
        self._configure_parameters(dp.params)

        # Set up callbacks using lambdify
        time = self.dp.eqn_env.t
        inputs = [s.s for s in self.dp.diagram.input_syms]
        params = [s.s for s in self.dp.params.keys()]
        self.n_ode = sed.n_ode
        self.n_alg = sed.n_alg
        sym_args = (time, sed.x, sed.y, *inputs, *params)

        # Continuous State
        self._configure_continuous_state(sym_args, leaf_backend)

        # Output Ports
        if sed.is_scaled:
            outp_exprs = {
                sym: expr.subs(sed.vars_to_scaled_vars)
                for sym, expr in dp.outp_exprs.items()
            }
        else:
            outp_exprs = dp.outp_exprs
        self._configure_outputs(sym_args, outp_exprs, leaf_backend)

        # Zero Crossing
        if sed.is_scaled:
            zcs = {}
            for idx, zc_tuple in dp.zcs.items():
                zc_expr, direction, is_bool_expr = zc_tuple
                zcs[idx] = (
                    zc_expr.subs(sed.vars_to_scaled_vars),
                    direction,
                    is_bool_expr,
                )
        else:
            zcs = dp.zcs

        self._configure_zcs(sym_args, zcs, leaf_backend)

    def _configure_inputs(self, input_syms):
        # this ensure that inports of the acausal_system are in the same order as
        # the 'inputs' portion of the lambdify args
        insym_to_portid = {}
        for sym in input_syms:
            idx = self.declare_input_port(name=sym.name)
            insym_to_portid[sym] = idx

        self.insym_to_portid = insym_to_portid

    def initialize(self, *args, **kwargs):
        # presently, this is unused.
        # resolved_args = [
        #     arg.get() if isinstance(arg, Parameter) else arg for arg in args
        # ]
        if args:
            self.dp._update_dpd()
            raise AcausalCompilerError(
                message="AcausalSystem initialize method detected unamed args. This is not supported.",
                dpd=self.dp.dpd,
            )

        # all the acausal component params are passed in through kwargs
        resolved_kwargs = {
            k: kwarg.get() if isinstance(kwarg, Parameter) else kwarg
            for k, kwarg in kwargs.items()
        }
        for k, v in self.dp.params.items():
            if k.validator:
                # not all params have validation, so skip if the validator is None.
                resolved_val = resolved_kwargs[k.name]
                if not k.validator(resolved_val):
                    raise AcausalModelError(k.invalid_msg)

    def _configure_parameters(self, params):
        for k, v in params.items():
            self.declare_dynamic_parameter(k.name, v)

    def _configure_continuous_state(self, sym_args, leaf_backend):
        sp_rhs = sp.lambdify(
            sym_args,
            self.sed.f + self.sed.g,
            modules=[leaf_backend, {"cnp": cnp}],
        )
        mass_matrix = np.concatenate((np.ones(self.n_ode), np.zeros(self.n_alg)))

        def _rhs(time, state, *u, **params):
            cstate = state.continuous_state
            param_values = [params[str(k)] for k in self.dp.params.keys()]
            x = cstate[: self.n_ode]
            y = cstate[self.n_ode :]  # noqa
            return cnp.array(sp_rhs(time, x, y, *u, *param_values))

        self.declare_continuous_state(
            shape=(self.n_ode + self.n_alg), ode=_rhs, mass_matrix=mass_matrix
        )

    def _configure_outputs(self, sym_args, outp_exprs, leaf_backend):
        outsym_to_portid = {}
        if self.dp.diagram.num_outputs == 0:
            # if not output, output the state vector
            self.declare_continuous_state_output(name=f"{self.name}:output")
            outsym_to_portid = None
        else:

            def _make_outp_callback(outp_expr):
                lambdify_output = sp.lambdify(
                    sym_args,
                    outp_expr,
                    modules=[leaf_backend, {"cnp": cnp}],
                )

                def _output_fun(time, state, *u, **params):
                    cstate = state.continuous_state
                    param_values = [params[str(k)] for k in self.dp.params.keys()]
                    x, y = cstate[: self.n_ode], cstate[self.n_ode :]  # noqa
                    return cnp.array(lambdify_output(time, x, y, *u, *param_values))

                return _output_fun

            # declaring acausal_system output ports in this order means that the ordering
            # 'source of truth' is self.model.output_syms which can be used to link
            # back to the acausal sensors causal port for diagram link src point remapping.
            for sym, outp_expr in outp_exprs.items():
                _output = _make_outp_callback(outp_expr)
                idx = self.declare_output_port(
                    _output,
                    name=sym.name,  # FIXME: not the name from the block port
                    prerequisites_of_calc=[DependencyTicket.xc],
                    requires_inputs=True,
                )
                outsym_to_portid[sym] = idx

        self.outsym_to_portid = outsym_to_portid

    def _configure_zcs(self, sym_args, zcs, leaf_backend):
        def _make_zc_callback(zc_expr, is_bool_expr):
            lambdify_zc = sp.lambdify(
                sym_args,
                zc_expr,
                modules=[leaf_backend, {"cnp": cnp}],
            )

            if is_bool_expr:
                # zero crossing are always expecting a float, so when the zero crossing condition
                # is defined by a boolean, we have this extra cnp.where() function whihc maps it
                # to a float. the same is done in the IfThenEsle block.
                def _zc_fun(time, state, *u, **params):
                    cstate = state.continuous_state
                    param_values = [params[str(k)] for k in self.dp.params.keys()]
                    x, y = cstate[: self.n_ode], cstate[self.n_ode :]  # noqa
                    return cnp.where(
                        cnp.array(lambdify_zc(time, x, y, *u, *param_values)), 1.0, -1.0
                    )

            else:

                def _zc_fun(time, state, *u, **params):
                    cstate = state.continuous_state
                    param_values = [params[str(k)] for k in self.dp.params.keys()]
                    x, y = cstate[: self.n_ode], cstate[self.n_ode :]  # noqa
                    return cnp.array(lambdify_zc(time, x, y, *u, *param_values))

            return _zc_fun

        # just copying what we do for outputs.
        for idx, zc_tuple in zcs.items():
            zc_expr, direction, is_bool_expr = zc_tuple
            _zc = _make_zc_callback(zc_expr, is_bool_expr)
            self.declare_zero_crossing(_zc, direction=direction)

    def initialize_static_data(self, context):
        dp = self.dp
        sed = self.sed
        try:
            u = self.collect_inputs(context)
            knowns_new = {}
            next_input_idx = 0
            for known in sed.knowns.keys():
                sym = dp.syms_map[known]
                if sym.kind == SymKind.inp:
                    # FIXME: we are relying on the fact that inputs are always
                    # appearing in the same order, which they are, but this is
                    # not robust going forward. we should re-architect where
                    # initial conditions computation input data is prepared,
                    # such that we dont have to partially compute it, and then
                    # go back and update it at this point.
                    knowns_new[known] = u[next_input_idx]
                    next_input_idx = next_input_idx + 1

            knowns = sed.knowns.copy()
            knowns.update(knowns_new)

            X_ic_mapping = compute_initial_conditions(
                sed.t, sed.eqs, sed.X, sed.ics, sed.ics_weak, knowns, verbose=True
            )

            x_ic = [X_ic_mapping[sed.dae_X_to_X_mapping[var]] for var in sed.x]
            y_ic = [X_ic_mapping[sed.dae_X_to_X_mapping[var]] for var in sed.y]

            if sed.is_scaled:
                x_ic = [val / sed.Ss[idx] for idx, val in enumerate(x_ic)]
                y_ic = [val / sed.Ss[idx + sed.n_ode] for idx, val in enumerate(y_ic)]

            x0 = np.array(x_ic + y_ic, dtype=float)

            self._default_continuous_state = x0
            local_context = context[self.system_id].with_continuous_state(x0)
            context = context.with_subcontext(self.system_id, local_context)

        except UpstreamEvalError:
            print(
                "DerivativeDiscrete.initialize_static_data: UpstreamEvalError. "
                "Continuing without default value initialization."
            )
        return super().initialize_static_data(context)


class AcausalCompiler:
    """
    This class ochestrates the compilation of Acausal models to Acausal Phleafs.

    There are 3 primary stages:
        1] diagram_processing. AcausalDiagram -> DAEs
        2] index_reduction. DAEs -> index-1 DAEs
        3] generate_acausal_system. index-1 DAEs -> pleaf
    """

    def __init__(
        self,
        eqn_env: EqnEnv,
        diagram: AcausalDiagram,
        scale: bool = False,
        verbose: bool = False,
    ):
        self.dp = DiagramProcessing(
            eqn_env,
            diagram,
            verbose=verbose,
        )
        self.index_reduction_done = False
        self.scale = scale
        self.verbose = verbose

        build_recorder.create_acausal_compiler()

    def diagram_processing(self):
        self.dp()

    def index_reduction(self):
        self.ir = IndexReduction(
            ir_inputs_from_dp=self.dp.index_reduction_inputs,
            dpd=self.dp.dpd,
            verbose=self.verbose,
        )
        self.sed = self.ir(scale=self.scale)
        self.index_reduction_done = True

    def generate_acausal_system(
        self,
        name="acausal_system",
        leaf_backend="jax",
    ):
        """
        This function is used for generating AcasualSystem from an AcausalDiagram.
        """
        if not self.dp.diagram_processing_done:
            self.diagram_processing()
        if not self.index_reduction_done:
            # NOTE: presently in from_model_json.py, self.index_reduction_done is set to True
            # this is only to temporarily skip index reduction when testing from json.
            self.index_reduction()

        with build_recorder.paused():
            system = AcausalSystem(self.dp, self.sed, name, leaf_backend)
        build_recorder.compile_acausal_diagram(system)

        return system

    # execute compilation
    def __call__(self, name="acausal_system", leaf_backend="jax", return_sed=False):
        self.diagram_processing()
        self.index_reduction()
        system = self.generate_acausal_system(name=name, leaf_backend=leaf_backend)
        if return_sed:
            return system, self.sed
        return system
