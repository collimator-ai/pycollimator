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

from dataclasses import dataclass
import itertools
import warnings
from typing import TYPE_CHECKING, Optional
from ..types import DiagramProcessingData, IndexReductionInputs
from ..error import AcausalModelError
import jax
from collimator.backend.typing import ArrayLike
from collimator.lazy_loader import LazyLoader
from collimator.backend import numpy_api as cnp
from collimator.backend.typing import Array

from .equation_utils import (
    compute_initial_conditions,
    extract_vars,
    order_vars_by_impact,
    process_equations,
    compute_condition_number,
)
from .graph_utils import (
    augmentpath,
    delete_var_nodes_with_zero_A,
    is_structurally_feasible,
    sort_block_by_number_of_eq_derivatives,
)

if TYPE_CHECKING:
    import networkx as nx
    import sympy as sp
    from networkx.algorithms import bipartite
else:
    sp = LazyLoader("sp", globals(), "sympy")
    nx = LazyLoader("nx", globals(), "networkx")
    bipartite = LazyLoader("bipartite", globals(), "networkx.algorithms.bipartite")


@dataclass
class SemiExplicitDAE:
    """
    Dataclass to store the semi-explicit form of the DAE system obtained after
    index reduction and *dummy derivatives* substitution.

    ```
    ẋ = f(t,x,y,θ)
    g(t,x,y,θ)=0
    ```

    It also stores:
    1. the mapping `dae_X_to_X_mapping` that maps the variables `x, ẋ, y` (dae_X) in the
    above equations, which may be dummy derivatives, to the corresponding original
    variables `X`.
    2. the variables `eqs, X, ics, ics_weak` that need to be solved for the
    computation of consistent initial conditions. This is needed because the IC
    computation needs to happen during Acausal system creation to enable changing of
    parameters (knowns `θ`).
    3. Variables to indicate whether variables are scaled; scaling factors if
    scaling is done; and a substitution mapping for ZCs and output expressions
    """

    # variables for semi-explicit DAE system
    t: sp.Symbol
    x: list
    x_dot: list  # ẋ
    y: list
    f: list
    g: list
    knowns: dict  # θ
    n_ode: int
    n_alg: int

    # map from dummy derivative variables to original variables
    dae_X_to_X_mapping: dict

    # variables for consistent initialization
    eqs: list
    X: list
    ics: dict
    ics_weak: dict

    # bool to indicated whether variables have been scaled
    # the state variables are s = [x,y]
    # s_scaled[i] = s[i]/Ss[i]
    is_scaled: bool
    Ss: Array  # the array of scaling factors
    vars_to_scaled_vars: dict
    scaled_vars_to_vars: dict


class IndexReduction:
    """
    Class to perform index reduction of a DAE system using the Pantelides algorithm
    and the method of dummy derivatives.

    Pantelides, C.C., 1988. The consistent initialization of differential-algebraic
    systems. SIAM Journal on scientific and statistical computing, 9(2), pp.213-231.

    Mattsson, S.E. and Söderlind, G., 1993. Index reduction in differential-algebraic
    equations using dummy derivatives. SIAM Journal on Scientific Computing, 14(3).

    Objects of this call can be initialized either by providing the necessary parameters
    `t, eqs, knowns, ics, ics_weak` directly or by passing an object `ir_inputs_from_dp`
    obtained from diagram processing steps.

    Parameters:
        t : sympy.Symbol
            The independent variable representing time.
        eqs : list of sympy.Expr
            The list of expressions (`expr`) in the DAE system. Each equation is assumed
            to be `expr=0`.
        knowns : dict
            Dictionary of known variables in the DAE system. The keys are the known
            sympy variables and the values are their corresponding numeric values.
        ics : dict
            Dictionary of (strong) initial conditions for the DAE system. The keys are
            the sympy variables and the values are the numerical initial values.
        ics_weak : dict
            Dictionary of weak initial conditions for the DAE system. The keys are the
            sympy variables and the values are the numerical initial values.
        ir_inputs_from_dp : IndexReductionInputs
            NamedTuple containing the inputs for index reduction obtained from
            DiagramProcessing.
        dpd: DiagramProcessingData
            NamedTuple containing the data obtained from DiagramProcessing. This is
            used to propagate errors upstream when `ir_inputs_from_dp` is used.
        verbose : bool
            Flag to print the intermediate steps of the index reduction pipeline.
    """

    def __init__(
        self,
        t: Optional[sp.Symbol] = None,
        eqs: Optional[list[sp.Expr]] = None,
        knowns: Optional[dict[sp.Expr, float]] = None,
        ics: Optional[dict[sp.Expr, float]] = None,
        ics_weak: Optional[dict[sp.Expr, float]] = None,
        ir_inputs_from_dp: Optional[IndexReductionInputs] = None,
        dpd: Optional[DiagramProcessingData] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.index_reduction_done = False
        self.ic_computed = False
        self.dpd = dpd
        if ir_inputs_from_dp is not None:
            # diagram_processing data is passed in as NamedTuple
            (
                self.t,
                self.x,
                self.x_dot,
                self.y,
                self.X,
                self.eqs,
                self.vars_in_eqs,
                self.eqs_idx_dict,
                self.knowns,
                self.known_vars,
                self.ics,
                self.ics_weak,
            ) = ir_inputs_from_dp
            self.eqs_idx_dp = list(self.eqs_idx_dict.values())
        else:
            self.t = t
            self.eqs = eqs
            self.eqs_idx_dp = list(range(len(eqs)))
            self.knowns = knowns
            self.known_vars = set(knowns.keys())
            self.ics = {} if ics is None else ics
            self.ics_weak = {} if ics_weak is None else ics_weak
            self.verbose = verbose

            (
                self.x,
                self.x_dot,
                self.y,
                self.X,
                self.vars_in_eqs,
                self.eqs_idx,
            ) = process_equations(self.eqs, self.known_vars)

        self.condition_number_threshold = 1e4

        # validate IC types, raise error if necessary.
        ics_with_invalid_types = {}
        for k, v in self.ics.items():
            if not isinstance(v, ArrayLike):
                ics_with_invalid_types[k] = type(v)
        for k, v in self.ics_weak.items():
            if not isinstance(v, ArrayLike):
                ics_with_invalid_types[k] = type(v)
        if ics_with_invalid_types:
            invalid_types = set(ics_with_invalid_types.values())
            types_str = "".join(f"{typ}, " for typ in list(invalid_types))
            message = "Invalid types detected in initial conditions." + types_str + "."
            raise AcausalModelError(
                message=message,
                dpd=self.dpd,
                variables=ics_with_invalid_types.keys(),
            )

    def __call__(self, scale: bool = False):
        self.check_system()
        self.prepare_pantelides_system()
        self.pre_pantelides_structural_singularity_check()
        self.pantelides()
        self.find_free_variables_for_consistent_initialization()
        self.initial_condition_check_and_tweaks()
        self.make_BLT_graph()
        self.dummy_derivatives()
        self.index_reduction_done = True
        self.convert_to_semi_explicit()

        if scale:
            X_ic_mapping = compute_initial_conditions(
                self.t,
                self.eqs,
                self.X,
                self.ics,
                self.ics_weak,
                self.knowns,
                verbose=self.verbose,
            )
            self.scale_se_equations(X_nom_mapping={}, X_ic_mapping=X_ic_mapping)
            sed = SemiExplicitDAE(
                self.t,
                self.ses_x,
                self.ses_x_dot,
                self.ses_y,
                self.ses_f,
                self.ses_g,
                self.knowns,
                len(self.ses_x),
                len(self.ses_y),
                self.ses_dae_X_to_X_mapping,
                self.eqs,
                self.X,
                self.ics,
                self.ics_weak,
                is_scaled=True,
                Ss=self.Ss,
                vars_to_scaled_vars=self.ses_vars_to_scaled_vars,
                scaled_vars_to_vars=self.ses_scaled_vars_to_vars,
            )
        else:
            sed = SemiExplicitDAE(
                self.t,
                self.se_x,
                self.se_x_dot,
                self.se_y,
                self.se_x_dot_rhs,
                self.se_alg_eqs,
                self.knowns,
                len(self.se_x),
                len(self.se_y),
                self.se_dae_X_to_X_mapping,
                self.eqs,
                self.X,
                self.ics,
                self.ics_weak,
                is_scaled=False,
                Ss=cnp.ones(len(self.se_x) + len(self.se_y)),
                vars_to_scaled_vars={},
                scaled_vars_to_vars={},
            )

        return sed

    def run_dev(self, config=None):
        """
        Only for development. Should not be called in production.
        """
        self.check_system()
        self.prepare_pantelides_system()
        self.pre_pantelides_structural_singularity_check()
        self.pantelides()
        self.find_free_variables_for_consistent_initialization()
        self.initial_condition_check_and_tweaks()
        self.X_ic_mapping = compute_initial_conditions(
            self.t,
            self.eqs,
            self.X,
            self.ics,
            self.ics_weak,
            self.knowns,
            config=config,
            verbose=self.verbose,
        )
        self.ic_computed = True
        self.make_BLT_graph()
        self.dummy_derivatives()

    def check_system(self):
        if self.y is None:
            warnings.warn("No algebraic variables found in the DAE system")
            self.is_ODE = True
        elif self.x is None:
            warnings.warn(
                "No differential variables exist. Consider using an algebraic solver."
            )

        self.n = len(self.x)
        self.m = len(self.y)

        self.N = self.n + self.m  # number of equations {x_dot, y}
        self.M = 2 * self.n + self.m  # number of variables (x, x_dot, y)

        if self.N != len(self.eqs):
            # Print system info before raising error
            print("##### Input system Information #####", "\n")
            print(f"Number of equations: {len(self.eqs)}")
            print(f"Number of differential variables: {self.n}")
            print(f"Number of algebraic variables: {self.m}")

            print("\n")
            for idx, eq in enumerate(self.eqs):
                print(f"Equation {idx} ({self.eqs_idx_dp[idx]}): {eq}")
            print("\n")
            for idx, x in enumerate(self.X):
                print(f"Variable {idx}: {x}")
            print("\n")

            print("Knowns:")
            for k, v in self.knowns.items():
                print(f"{k} = {v}")
            print("\n")

            print("Strong ICs provided for:")
            for k, v in self.ics.items():
                print(f"{k} = {v}")
            print("\n")
            print("Weak ICs provided for:")
            for k, v in self.ics_weak.items():
                print(f"{k} = {v}")
            print("\n")

            message = f"Mismatch between the number of equations {len(self.eqs)} and the number of variables {self.N}."
            raise AcausalModelError(message=message, dpd=self.dpd)

        if self.verbose:
            print("##### Input system Information #####", "\n")
            print(f"Total equations: {self.N}")
            print(f"Total variables: {self.M}")
            print(f"Number of differential variables: {self.n}")
            print(f"Number of algebraic variables: {self.m}")
            print("\n")
            for idx, eq in enumerate(self.eqs):
                print(f"Equation {idx} ({self.eqs_idx_dp[idx]}): {eq}")
            print("\n")
            for idx, x in enumerate(self.X):
                print(f"Variable {idx}: {x}")
            print("\n")

            print("Knowns:")
            for k, v in self.knowns.items():
                print(f"{k} = {v}")
            print("\n")

            print("Strong ICs provided for:")
            for k, v in self.ics.items():
                print(f"{k} = {v}")
            print("\n")
            print("Weak ICs provided for:")
            for k, v in self.ics_weak.items():
                print(f"{k} = {v}")
            print("\n")

        # Check if ics are provided in both strong and weak forms
        duplicate_ics = set(self.ics.keys()).intersection(set(self.ics_weak.keys()))
        if duplicate_ics:
            warnings.warn(
                f"Initial conditions provided for the following variables in both "
                f"strong and weak form: {duplicate_ics}. Weak forms for these "
                f"conditions will be ignored."
            )
            self.ics_weak = {
                k: v for k, v in self.ics_weak.items() if k not in duplicate_ics
            }

        # Check of the union of strong and weak ICs is X
        ics_missing = set(self.X) - (set(self.ics.keys()) | (set(self.ics_weak.keys())))
        if ics_missing:
            # FIXME: decide whetere to raise error or set missing ICs to zero
            raise_error = False

            if raise_error:
                raise ValueError(
                    "Initial conditions should be provided for all the variables in the "
                    f"DAE system. The following {len(ics_missing)} variables are missing "
                    f"initial conditions {ics_missing}"
                )
            else:
                self.ics_weak.update({k: 0.0 for k in ics_missing})

    def prepare_pantelides_system(self):
        self.Nprime = self.N
        self.Mprime = self.M

        self.create_bipartite_graph()

        # Create association list
        A = [None] * len(self.X)
        for idx, x in enumerate(self.X):
            if sp.diff(x, self.t) in self.X:
                A[idx] = self.X.index(sp.diff(x, self.t))
        self.A = A

        self.assign = [None] * len(self.X)
        self.B = [None] * self.N

    def create_bipartite_graph(self):
        """
        Create a bipartite graph from the DAE system equations and variables.
        - Equation nodes named by their indices in self.eqs from 0 to N-1.
        - Variable nodes are named by their symbols in self.X. Forward and reverse
          mappings from these names to indices are created.
        """

        self.G = nx.Graph()

        # Add nodes with the bipartite attribute. Equation nodes are bipartite 0, and
        # variable nodes are bipartite 1.
        self.G.add_nodes_from([i for i, _ in enumerate(self.eqs)], bipartite=0)
        self.G.add_nodes_from(self.X, bipartite=1)

        # Add edges based on variable presence in each equation
        for eq_idx, (_, vars_in_eq) in enumerate(self.vars_in_eqs.items()):
            for var in vars_in_eq:
                self.G.add_edge(eq_idx, var)

        self.e_nodes = [n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0]
        self.v_nodes = [n for n, d in self.G.nodes(data=True) if d["bipartite"] == 1]

        # Create a mapping from variable node labels to indices and vice versa
        self.v_mapping = {node: idx for idx, node in enumerate(self.v_nodes)}
        self.reverse_v_mapping = {idx: node for node, idx in self.v_mapping.items()}

        # Graph to keep track of equation differentiations
        self.eq_diff_graph = nx.DiGraph()
        self.eq_diff_graph.add_nodes_from(self.e_nodes)

    def pre_pantelides_structural_singularity_check(self):
        """
        Check if the DAE system (any index) is structurally singular. This is done
        by adding `n` extra equations relating the differential variables `x`
        to their derivatives `x_dot`. This extended system of `2n+m`
        equations and `2n+m` variables is then analyzed for structural singularity.
        """
        G = self.G.copy()

        # Add `n` extra equations relating the the differential variables to their
        # derivatives
        for idx, x in enumerate(self.x):
            new_eq_idx = self.N + idx
            G.add_node(new_eq_idx, bipartite=0)
            G.add_edge(new_eq_idx, x)
            G.add_edge(new_eq_idx, sp.diff(x, self.t))

        # Find maximum matching
        mm = bipartite.matching.maximum_matching(G, top_nodes=self.v_nodes)

        if len(mm) != 2 * (2 * self.n + self.m):
            # raise ValueError(
            #     "The system of equations is structurally singular. The DAE system "
            #     "is ill-posed. Aborting!"
            # )
            message = "The system of equations is structurally singular. The DAE system is ill-posed. Aborting!"
            raise AcausalModelError(message=message, dpd=self.dpd)

    def pantelides(self, max_steps=20):
        """
        Algorithm 4.1 of
        Pantelides, C.C., 1988. The consistent initialization of differential-algebraic
        systems. SIAM Journal on scientific and statistical computing, 9(2), pp.213-231.
        """
        # Steps 1 and 2 are performed in `prepare_pantelides_system`
        # Step 3
        for k in range(self.Nprime):
            i = k
            pathfound = False
            counter_steps = 0
            while not pathfound and (counter_steps < max_steps):
                G = self.G.copy()
                delete_var_nodes_with_zero_A(G, self.A, self.X)
                nx.set_node_attributes(G, "white", "color")
                pathfound = False
                pathfound, self.assign = augmentpath(
                    G, i, pathfound, self.assign, self.v_mapping
                )
                colored_e_nodes = [
                    n
                    for n, d in G.nodes(data=True)
                    if d["color"] == "red" and d["bipartite"] == 0
                ]
                colored_v_nodes = [
                    n
                    for n, d in G.nodes(data=True)
                    if d["color"] == "red" and d["bipartite"] == 1
                ]

                if not pathfound:
                    # (i)
                    for v_node in colored_v_nodes:
                        j = self.v_mapping[v_node]
                        self.M = self.M + 1

                        new_diff_var = sp.diff(self.X[j], self.t)
                        self.X.append(new_diff_var)
                        self.G.add_node(new_diff_var, bipartite=1)
                        self.A.append(None)
                        self.assign.append(None)

                        self.v_nodes.append(new_diff_var)
                        self.v_mapping[new_diff_var] = (
                            self.M - 1
                        )  # -1 because of 0-based indexing
                        self.reverse_v_mapping[self.M - 1] = new_diff_var

                        # FIXME: when a new variable is introduced as a result
                        # of differentiating equations during index reduction,
                        # the weak IC for new variable is set to zero.
                        self.ics_weak[new_diff_var] = 0.0

                        self.A[j] = self.M - 1

                    # (ii)
                    for e_node in colored_e_nodes:
                        self.N = self.N + 1

                        new_eq_node = self.N - 1  # -1 because of 0-based indexing
                        self.G.add_node(new_eq_node, bipartite=0)
                        self.B.append(None)
                        self.eqs.append(sp.diff(self.eqs[e_node], self.t))

                        self.e_nodes.append(new_eq_node)

                        self.eq_diff_graph.add_node(self.N - 1)
                        self.eq_diff_graph.add_edge(e_node, self.N - 1)

                        neighbors = self.G.neighbors(e_node)
                        for v_node in neighbors:
                            j = self.v_mapping[v_node]
                            self.G.add_edge(new_eq_node, v_node)
                            if self.A[j] is not None:
                                self.G.add_edge(
                                    new_eq_node, self.reverse_v_mapping[self.A[j]]
                                )

                        self.B[e_node] = self.N - 1

                        # add to dp eq index
                        self.eqs_idx_dp.append(-new_eq_node)

                    # (iii)
                    for v_node in colored_v_nodes:
                        j = self.v_mapping[v_node]
                        self.assign[self.A[j]] = self.B[self.assign[j]]

                    # (iv)
                    i = self.B[i]
                counter_steps += 1

        # Variable to equation matching: index in self.X -> index in self.eqs
        self.matching = {}
        self.reverse_matching = {}

        for idx_var, idx_eq in enumerate(self.assign):
            if idx_eq is not None:
                self.matching[idx_var] = idx_eq
                self.reverse_matching[idx_eq] = idx_var

        self.pantelides_dae_eqs = [
            eq_idx
            for eq_idx in self.eq_diff_graph.nodes()
            if self.eq_diff_graph.out_degree(eq_idx) == 0
        ]

        self.pantelides_dae_vars = [
            self.reverse_matching[eq_idx] for eq_idx in self.pantelides_dae_eqs
        ]

        self.pantelides_dae_reverse_matching = {
            eq_idx: self.reverse_matching[eq_idx] for eq_idx in self.pantelides_dae_eqs
        }

        if self.verbose:
            assignment_dict = dict(
                zip(self.X, [i if i is not None else "" for i in self.assign])
            )
            eq_differentiation_dict = dict(
                zip(
                    [i if i is not None else "" for i in range(len(self.B))],
                    [i if i is not None else "" for i in self.B],
                )
            )

            derivative_mapping_dict = {
                self.X[base]: self.X[derivative]
                for base, derivative in enumerate(self.A)
                if derivative is not None
            }

            print("##### Panteides Algorithm Completed #####", "\n")

            print(f"Total equations (before|after): {self.Nprime}|{self.N}")
            print(f"Total variables (before|after): {self.Mprime}|{self.M}")

            print("\n")
            print("# Variables", "\n")
            for idx, var in enumerate(self.X):
                print(f"Variable {idx}: {var}")
            print("\n")
            print("# Equations", "\n")
            for idx, eq in enumerate(self.eqs):
                print(f"Equation {idx} ({self.eqs_idx_dp[idx]}): {eq}")

            print("\n")
            print("# Variable assignments")
            for k, v in assignment_dict.items():
                print(
                    f"Variable {k} is assigned to -> e{v} ({self.eqs_idx_dp[v] if v else ''})"
                )

            print("\n")
            print("# Differentiated equations")
            for k, v in eq_differentiation_dict.items():
                print(
                    f"Differentiate e{k} ({self.eqs_idx_dp[k]}) to get  -> e{v} ({self.eqs_idx_dp[v] if v else ''})"
                )

            print("\n")
            print("# Derivatives present in the variable association list")
            for k, v in derivative_mapping_dict.items():
                print(f"Present derivative of {k} is  -> {v}")

            print("\n")
            print("##### Index-1 (atmost) system after Pantelides #####", "\n")

            print("# Equations in the index-1 DAE system", "\n")
            for eq_idx in self.pantelides_dae_eqs:
                print(
                    f"Equation {eq_idx} ({self.eqs_idx_dp[eq_idx]}): {self.eqs[eq_idx]}"
                )

            print("\n")
            print("Variables in the index-1 DAE system", "\n")
            for var_idx in self.pantelides_dae_vars:
                print(f"Variable {var_idx}: {self.X[var_idx]}")

    def find_free_variables_for_consistent_initialization(self):
        """
        Find the variables that can be freely chosen for consistent initialization
        of the system produced by the Pantelides algorithm.

        On completion, this method will create two attributes:
            - `X_free`: the variables in X that can be freely chosen for consistent
              initialization, and
            - `X_free_idx`: the indices of the X_free variables in self.X

        Method from the following reference (see Fig 7):
        R. W. H. SARGENT, The decomposition ofsystems ofprocedures and algebraic
        equations, in Numerical Analysis--Proceedings, Dundee, G. A. Watson, 1977,
        Lecture Notes in Mathematics 630, Springer- Verlag, Berlin, New York,
        Heidelberg, 1978.
        """
        # Determine if system is over/under-determined
        if self.N > self.M:
            # If Pantelides converges, we should never get here.
            raise ValueError(
                "The system of equations is over-determined. After Pantelides "
                f"algorithm there are {self.N} equations in {self.M} variabkes. "
                "Aborting!"
            )
        elif self.N == self.M:
            print(
                "\n"
                f"Structural analysis: {self.N} equations in {self.M} variables. "
                "No variables can be freely chosen for consistent initialization."
            )
            self.X_free_idx = []
            self.X_free = []
            return
        else:
            self.num_fake_equations = self.M - self.N

        # Create a new graph with the fake equations
        G = self.G.copy()
        e_nodes = self.e_nodes.copy()

        for i in range(self.num_fake_equations):
            new_eq_node = self.N + i
            G.add_node(new_eq_node, bipartite=0)
            e_nodes.append(new_eq_node)

            for v_node in self.v_nodes:
                G.add_edge(new_eq_node, v_node)

        # Find maximum matching
        _mm = bipartite.matching.maximum_matching(G, top_nodes=self.v_nodes)
        mm = {k: v for k, v in _mm.items() if k in e_nodes}

        if len(mm) != len(self.X):
            raise ValueError(
                "The system of equations for initial conditions obtained after "
                "Pantelides algorithm is not solvable. A complete matching could not "
                "be found."
            )
        # Variable to equation matching
        max_matching = {}
        # Equation to variable matching
        reverse_max_matching = {}

        for eq, var in mm.items():
            max_matching[self.v_mapping[var]] = eq
            reverse_max_matching[eq] = self.v_mapping[var]

        D = nx.DiGraph()
        D.add_nodes_from(reverse_max_matching.keys())

        for eq_parent, idx_matched_var in reverse_max_matching.items():
            eq_neighbors_of_matched_var = list(
                G.neighbors(self.reverse_v_mapping[idx_matched_var])
            )
            for eq_child in eq_neighbors_of_matched_var:
                if eq_child != eq_parent:
                    D.add_edge(eq_parent, eq_child)

        scc = [list(x) for x in nx.strongly_connected_components(D)]

        # Construct new graph with SCCs as nodes
        scc_graph = nx.DiGraph()
        scc_map = {}  # Map each node to its SCC
        for idx, component in enumerate(scc):
            scc_graph.add_node(idx)
            for node in component:
                scc_map[node] = idx

        # Add edges between SCCs in the new graph
        for u, v in D.edges():
            if scc_map[u] != scc_map[v]:
                scc_graph.add_edge(scc_map[u], scc_map[v])

        # Sort the SCCs in topological order
        topological_sorted_scc = list(nx.topological_sort(scc_graph))

        # Convert back to the actual nodes in the original graph
        eBLT = [scc[idx] for idx in topological_sorted_scc]

        last_block = eBLT[-1]

        self.X_free_idx = [reverse_max_matching[eq_idx] for eq_idx in last_block]
        self.X_free = [self.X[idx] for idx in self.X_free_idx]
        self.num_free_vars = len(self.X_free)
        if self.verbose:
            print("##### Structural analysis for consistent initialization #####", "\n")
            print("# Blocks Triangular equations")
            print(f"{eBLT=}", "\n")
            print(
                f"Structural analysis: any {self.M-self.N} variables from the following "
                f"{len(self.X_free)} variables can be freely chosen for consistent "
                f"initialization of {self.M} equations:",
                "\n",
            )
            for idx, x in enumerate(self.X_free):
                print(f"Variable {idx}: {x}")

    def handle_overdetermined_ics(self, num_ics_to_remove):
        """
        The number of strong ICs is larger than the number of free variables
        that can be chosen for consistent initialization.

        1. Order all the variables for which strong ICs are specified based on their
        impact on the Jacobian of the equations at `t=0`.
        2. From this ordered list of variables, select the variables which
            (i) yield a structurally nonsingular system,
            (ii) yield a condition number of the Jacobian lower than a threshold, and
            (iii) have the the lowest impact.
        """
        ordered_ics = order_vars_by_impact(
            self.t, self.eqs, self.X, self.ics, self.ics_weak, self.knowns
        )

        structurally_feasible_set_found = False
        numerically_feasible_set_found = False
        for removal_keys in itertools.combinations(
            ordered_ics.keys(), num_ics_to_remove
        ):
            ics = {k: v for k, v in self.ics.items() if k not in removal_keys}
            ics_weak = {
                **self.ics_weak,
                **{k: v for k, v in self.ics.items() if k in removal_keys},
            }

            if is_structurally_feasible(ics, self.G):
                structurally_feasible_set_found = True
                condition_number = compute_condition_number(
                    self.t, self.eqs, self.X, ics, ics_weak, self.knowns
                )

                if condition_number < self.condition_number_threshold:
                    self.ics = ics
                    self.ics_weak = ics_weak
                    numerically_feasible_set_found = True
                    print("Strong ICs moved to weak ICs for the following vars:")
                    for k in removal_keys:
                        print(f"{k}={self.ics_weak[k]}")
                    return

        if not structurally_feasible_set_found:
            raise ValueError(
                "A combination of initial conditions leading to a structurally "
                " feasible system could not be found. Aborting!"
            )
        if not numerically_feasible_set_found:
            warnings.warn(
                "A combination of initial conditions leading to a structurally "
                "feasible system was found, but a numerically feasible solution "
                "could not be found."
            )
        # if not found_solution:
        #     warnings.warn(
        #         "No removal of specified strong initial conditions resulted in a "
        #         " well conditioned Jacobian at `t=0`. Proceeding with removing the "
        #         " variables with lowest impact."
        #     )
        #
        #     removal_keys = [
        #         k for i, k in enumerate(ordered_ics.keys()) if i < num_ics_to_remove
        #     ]
        #
        #     self.ics = {k: v for k, v in self.ics.items() if k not in removal_keys}
        #     self.ics_weak = {
        #         **self.ics_weak,
        #         **{k: v for k, v in self.ics.items() if k in removal_keys},
        #     }

    def handle_determined_ics(self):
        """
        The number of strong ICs is equal to the number of free variables
        that can be chosen for consistent initialization.

        1. If structurally feasible, check condition number, raise a warning if the
        condition number is greater than a threshold, but proceed anyways.
        2. If not strcuturally infeasible, swap the vars (1 var, 2 vars, 3 vars ...
        sequentially) in strong ics with vars in X_free not in ics, until a structurally
        and numerically feasible solution is found.

        """
        if not is_structurally_feasible(self.ics, self.G):
            warnings.warn(
                "The number of initial conditions provided are correct. However, "
                "the provided set does not lead to a structurally feasible system. "
                f"Provided initial conditions were for variables: {self.ics.keys()}. "
                f"A different combination of initial values for any "
                f"{self.M-self.N} variables from "
                f"the following {len(self.X_free)} variables: {self.X_free} "
                f"will be selected. The combination will attempt to keep as many "
                f"original initial conditions as possible."
            )

            X_excluding_ics = [x for x in self.X if x not in self.ics.keys()]
            X_available_for_ic_swap = [x for x in X_excluding_ics if x in self.X_free]

            max_ics_swappable = min(len(X_available_for_ic_swap), len(self.ics))

            structurally_feasible_set_found = False
            numerically_feasible_set_found = False
            for swap_num_ics in range(max_ics_swappable):
                for swap_orig_ics in itertools.combinations(
                    self.ics.keys(), swap_num_ics
                ):
                    for swap_new_ics in itertools.combinations(
                        X_available_for_ic_swap, swap_num_ics
                    ):
                        ics = self.ics.copy()
                        ics_weak = self.ics_weak.copy()

                        # move swap_orig_ics from ics to ics_weak
                        ics_weak.update({k: ics.pop(k) for k in swap_orig_ics})

                        # add swap_new_ics to ics and remove from ics_weak
                        ics.update({k: ics_weak.pop(k) for k in swap_new_ics})

                        if is_structurally_feasible(ics, self.G):
                            structurally_feasible_set_found = True
                            condition_number = compute_condition_number(
                                self.t, self.eqs, self.X, ics, ics_weak, self.knowns
                            )

                            if condition_number < self.condition_number_threshold:
                                self.ics = ics
                                self.ics_weak = ics_weak
                                numerically_feasible_set_found = True
                                print(
                                    "The following strong ICs were moved to weak ICs:"
                                )
                                for k in swap_orig_ics:
                                    print(f"{k}={self.ics_weak[k]}")
                                print(
                                    "in lieu of which, the following weak ICs were "
                                    "moved to strong ICs:"
                                )
                                for k in swap_new_ics:
                                    print(f"{k}={self.ics[k]}")
                                return

            if not structurally_feasible_set_found:
                raise ValueError(
                    "A combination of initial conditions leading to a structurally "
                    " feasible system could not be found. Aborting!"
                )
            if not numerically_feasible_set_found:
                warnings.warn(
                    "A combination of initial conditions leading to a structurally "
                    "feasible system was found, but a numerically feasible solution "
                    "could not be found"
                )

        condition_number = compute_condition_number(
            self.t, self.eqs, self.X, self.ics, self.ics_weak, self.knowns
        )

        if condition_number > self.condition_number_threshold:
            warnings.warn(
                "The specified strong initial conditions result in an ill conditioned "
                " Jacobian at `t=0`. Proceeding anyways."
            )

    def handle_underdetermined_ics(self, num_new_ics):
        """
        The number of strong ICs is smaller than the number of free variables
        that can be chosen for consistent initialization.

        1. Sequentially go through all possible combinations (in groups of
        `num_new_ics`) of the free variables not in current ics.
        2. For combinations that yield a structurally nonsingular system, compute
            the condition number of the Jacobian at `t=0`.
        3. Sort the combinations based on the condition number and select the
            combination with the lowest condition number.

        """
        acceptable_weak_ic_vars = []
        acceptable_weak_ic_vars_idxs = []
        condition_numbers = []

        X_excluding_ics = [x for x in self.X if x not in self.ics.keys()]
        X_availale_for_new_ics = [x for x in X_excluding_ics if x in self.X_free]

        X_availale_for_new_ics_idx = [self.X.index(x) for x in X_availale_for_new_ics]

        for x_chosen_idxs in itertools.combinations(
            X_availale_for_new_ics_idx, num_new_ics
        ):
            x_chosen = [self.X[idx] for idx in x_chosen_idxs]

            ics = self.ics.copy()
            ics_weak = self.ics_weak.copy()
            ics.update({var: ics_weak.pop(var) for var in x_chosen})

            if is_structurally_feasible(ics, self.G):
                condition_number = compute_condition_number(
                    self.t, self.eqs, self.X, ics, ics_weak, self.knowns
                )
                acceptable_weak_ic_vars.append(x_chosen)
                acceptable_weak_ic_vars_idxs.append(x_chosen_idxs)
                condition_numbers.append(condition_number)

        if not acceptable_weak_ic_vars:
            raise ValueError(
                "A combination of initial conditions leading to a structurally "
                " feasible system could not be found. Aborting!"
            )

        if sorted(condition_numbers)[0] > self.condition_number_threshold:
            warnings.warn(
                "A combination of initial conditions leading to a structurally "
                "feasible system was found, but a numerically feasible solution "
                "could not be found."
            )

        # sort based on condition numbers
        sorted_idxs = [
            idx for idx, _ in sorted(enumerate(condition_numbers), key=lambda x: x[1])
        ]
        self.acceptable_weak_ic_vars = [acceptable_weak_ic_vars[i] for i in sorted_idxs]
        self.acceptable_weak_ic_vars_idxs = [
            acceptable_weak_ic_vars_idxs[i] for i in sorted_idxs
        ]
        self.condition_numbers = [condition_numbers[i] for i in sorted_idxs]

        self.ics_new = {
            var: self.ics_weak.pop(var) for var in self.acceptable_weak_ic_vars[0]
        }
        self.ics = {**self.ics, **self.ics_new}

        print("The following weaks ICs were converted to strong ICs:")
        for k, v in self.ics_new.items():
            print(f"{k}={v}")

    def initial_condition_check_and_tweaks(self):
        """
        This method analyzes the combination of strong and weak initial conditions
        with structural and numerical analysis.

        - Over-determined: If the number of strong ICs is greater than the number of
          free variables, the strong ICs with the least impact and yet yielding a
          structurally and numerically nonsingular IC system are produced.
        - Determined: If the number of strong ICs is equal to the number of free
          variables, the IC system is analyzed for structural and numerical
          singularity.
        - Under-determined: If the number of strong ICs is less than the number of
         free variables, the free variables are sequentially combined and analyzed for
         structural and numerical singularity. The combination with the lowest condition
         number is chosen.

        After this procedure, the attributes `self.ics` and `self.ics_weak` will
        reflect the finaally chosen set of strong and weak ICs.
        """

        # the strong ICs must be a subset of the free variables, otherwise they have
        # no impact apart from starting guess.

        # Find the intersection of ics and X_free
        self.ics_not_in_X_free = {
            k: v for k, v in self.ics.items() if k not in self.X_free
        }
        self.ics = {k: v for k, v in self.ics.items() if k in self.X_free}
        self.ics_weak.update(self.ics_not_in_X_free)

        if self.ics_not_in_X_free:
            print(
                "\nStrong initial conditions for the following variables are not amongst "
                "the free variables. These initial conditions will be used as starting "
                "guesses only.\n"
            )
            for k, v in self.ics_not_in_X_free.items():
                print(f"{k} = {v}")

        # Over-determined system
        num_ics_required = self.M - self.N
        if len(self.ics) > num_ics_required:
            num_ics_to_remove = len(self.ics) - num_ics_required
            print(
                f"\nToo many strong initial conditions. Converting {num_ics_to_remove} "
                "of these to weak initial conditions.\n"
            )
            self.handle_overdetermined_ics(num_ics_to_remove)
        # Determined system
        elif len(self.ics) == num_ics_required:
            print("\nCorrect number of strong initial conditions.\n")
            self.handle_determined_ics()
        # under-determined system
        elif len(self.ics) < num_ics_required:
            print(
                "\nToo few strong initial conditions. "
                "Determining acceptable weak initial conditions.\n"
            )
            num_new_ics = num_ics_required - len(self.ics)
            self.handle_underdetermined_ics(num_new_ics)

        print("\nStrong ICs post tweaking for numerical computation:")
        for k, v in self.ics.items():
            print(f"{k} = {v}")
        print("\nWeak ICs post tweaking for numerical computation:")
        for k, v in self.ics_weak.items():
            print(f"{k} = {v}")
        print("\n")

    def make_BLT_graph(self):
        """
        For the atmost index-1 system produced by Pantelides algorith, create a
        Block lower triangular (BLT) ordering. The BLT ordering is a topological
        ordering of the strongly connected components (SCCs) of the directed graph.
        """

        # Create an equation dependency (in terms of matched variables) graph D
        D = nx.DiGraph()
        D.add_nodes_from(self.pantelides_dae_reverse_matching.keys())

        for eq_parent, idx_matched_var in self.pantelides_dae_reverse_matching.items():
            eq_neighbors_of_matched_var = list(
                self.G.neighbors(self.reverse_v_mapping[idx_matched_var])
            )
            for eq_child in eq_neighbors_of_matched_var:
                if eq_child != eq_parent:
                    D.add_edge(eq_parent, eq_child)

        scc = [list(x) for x in nx.strongly_connected_components(D)]

        # Construct new graph with SCCs as nodes
        scc_graph = nx.DiGraph()
        scc_map = {}  # Map each node to its SCC
        for idx, component in enumerate(scc):
            scc_graph.add_node(idx)
            for node in component:
                scc_map[node] = idx

        # Add edges between SCCs in the new graph
        for u, v in D.edges():
            if scc_map[u] != scc_map[v]:
                scc_graph.add_edge(scc_map[u], scc_map[v])

        # Sort the SCCs in topological order
        topological_sorted_scc = list(nx.topological_sort(scc_graph))

        # Convert back to the actual nodes in the original graph
        self.eBLT = [scc[idx] for idx in topological_sorted_scc]

        if self.verbose:
            print("##### Block Lower Triangular (BLT) ordering #####", "\n")

            print("BLT equation ordering")
            print(
                [
                    [f"e{idx} ({self.eqs_idx_dp[idx]})" for idx in block]
                    for block in self.eBLT
                ]
            )

            print("BLT variable ordering")
            print(
                [
                    [self.X[self.pantelides_dae_reverse_matching[idx]] for idx in block]
                    for block in self.eBLT
                ]
            )

    def dummy_derivatives(self):
        """
        Algorithm in Section 3.1 of
        Mattsson, S.E. and Söderlind, G., 1993. Index reduction in
        differential-algebraic equations using dummy derivatives.
        SIAM Journal on Scientific Computing, 14(3), pp.677-692.
        """
        BLT_eq_blocks = self.eBLT

        self.dummy_vars = {}
        self.final_dae_eqs_pre_replacement = []
        self.replace = {}

        for unsorted_eq_block in BLT_eq_blocks:
            # Step 1
            num_parents, eq_block = sort_block_by_number_of_eq_derivatives(
                self.eq_diff_graph, unsorted_eq_block
            )
            vars_block = [
                self.pantelides_dae_reverse_matching[eq_idx] for eq_idx in eq_block
            ]

            g = sp.Matrix([self.eqs[eq_idx] for eq_idx in eq_block])
            z = sp.Matrix([self.X[var_idx] for var_idx in vars_block])
            G = g.jacobian(z)

            block_replace = {}
            sub_blocks = [eq_block]
            while True:
                # Step 2
                if sum(num_parents) == 0:
                    # Go to Step 6
                    break
                else:
                    # Step 3
                    m = sum([1 for n in num_parents if n != 0])

                    H = G[:m, :]

                    # Step 4
                    _, pivot_columns = H.rref()

                    # Step 5
                    M = H[:, pivot_columns]

                    for replacing_eq, replacing_var in zip(
                        eq_block[:m], [vars_block[idx] for idx in pivot_columns]
                    ):
                        block_replace[replacing_eq] = replacing_var

                    G = M
                    eq_block = [
                        list(self.eq_diff_graph.predecessors(eq_idx))[0]
                        for eq_idx in eq_block[:m]
                    ]
                    vars_block = [
                        self.A.index(vars_block[idx]) for idx in pivot_columns
                    ]

                    num_parents = [n - 1 for n in num_parents[:m]]

                    sub_blocks.append(eq_block)

            # Step 6

            final_block_eqs = []
            if block_replace:
                # Create dummy variables
                block_dummy_vars = {}
                for eq_idx, var_idx in block_replace.items():
                    dummy_var = sp.Function("d_" + str(self.X[var_idx]))(self.t)
                    block_dummy_vars[self.X[var_idx]] = dummy_var
            else:
                block_dummy_vars = {}

            # Gather equations in reverse block order
            for sub_block in reversed(sub_blocks):
                for eq_idx in sub_block:
                    final_block_eqs.append(self.eqs[eq_idx])

            self.final_dae_eqs_pre_replacement.extend(final_block_eqs)
            self.dummy_vars.update(block_dummy_vars)
            self.replace.update(block_replace)

        # Replace true variables with dummy variables
        self.final_dae_eqs = []
        for eq in self.final_dae_eqs_pre_replacement:
            replaced_eq = eq.subs(self.dummy_vars)
            self.final_dae_eqs.append(replaced_eq)

        Xset = set(self.X)
        Dset = set(self.dummy_vars.keys())  # dummy vars are algebraic
        Vset = Xset - Dset

        self.final_dae_x, self.final_dae_y = set(), set()

        for var in Vset:
            if isinstance(var, sp.Derivative):
                self.final_dae_x.add(var.expr)
            else:
                self.final_dae_y.add(var)

        self.final_dae_y = self.final_dae_y.difference(self.final_dae_x)
        self.final_dae_y = self.final_dae_y.union(
            {self.dummy_vars[var] for var in Dset}
        )

        self.final_dae_x = list(self.final_dae_x)
        self.final_dae_y = list(self.final_dae_y)

        self.final_dae_x_dot = [var.diff(self.t) for var in self.final_dae_x]
        self.final_dae_X = list(
            set().union(self.final_dae_x, self.final_dae_x_dot, self.final_dae_y)
        )

        self.X_to_dae_X_mapping = {}
        for var in self.X:
            if var in self.dummy_vars:
                self.X_to_dae_X_mapping[var] = self.dummy_vars[var]
            else:
                self.X_to_dae_X_mapping[var] = var

        self.dae_X_to_X_mapping = {v: k for k, v in self.X_to_dae_X_mapping.items()}

        self.final_dae_system_is_purely_algebraic = len(self.final_dae_x) == 0

        if self.ic_computed:
            self.final_dae_x_ic = [
                self.X_ic_mapping[self.dae_X_to_X_mapping[var]]
                for var in self.final_dae_x
            ]
            self.final_dae_x_dot_ic = [
                self.X_ic_mapping[self.dae_X_to_X_mapping[var]]
                for var in self.final_dae_x_dot
            ]
            self.final_dae_y_ic = [
                self.X_ic_mapping[self.dae_X_to_X_mapping[var]]
                for var in self.final_dae_y
            ]

        if self.verbose:
            print("\n")
            print("#" * 10, "Dummy Derivatives computed succesfully", "#" * 10, "\n")
            print("#" * 10, "Final DAE equations F(x, x_dot, y)=0", "#" * 12, "\n")
            for idx, eq in enumerate(self.final_dae_eqs):
                print(f"Eq {idx:<4}:  ", eq)

            if self.ic_computed:
                print("\n# with, x =\n")
                for var, ic in zip(self.final_dae_x, self.final_dae_x_ic):
                    print(f"{str(var):30} with ic= {ic}")

                print("\n# x_dot =\n")
                for var, ic in zip(self.final_dae_x_dot, self.final_dae_x_dot_ic):
                    print(f"{str(var):30} with ic= {ic}")

                print("\n# and, y =\n")
                for var, ic in zip(self.final_dae_y, self.final_dae_y_ic):
                    print(f"{str(var):30} with ic= {ic}")

                print("\n", "#" * 60, "\n")
            else:
                print("\n# with, x =\n")
                for var in self.final_dae_x:
                    print(f"{str(var)}")

                print("\n# x_dot =\n")
                for var in self.final_dae_x_dot:
                    print(f"{str(var)}")

                print("\n# and, y =\n")
                for var in self.final_dae_y:
                    print(f"{str(var)}")

                print("\n", "#" * 60, "\n")

            if self.final_dae_system_is_purely_algebraic:
                print(
                    "The DAE system after dummy derivative substitution is purely "
                    "algebraic. Consider using an algebraic solver."
                )

    def convert_to_semi_explicit(self):
        """
        Convert the final DAE system after dummy derivatives to a semi-explicit form
        """

        if self.final_dae_system_is_purely_algebraic:
            # raise ValueError(
            #     "The DAE system after dummy derivative substitution is purely "
            #     "algebraic. Consider using an algebraic solver. Aborting conversion "
            #     "to semi-explicit form."
            # )
            message = (
                "The DAE system after dummy derivative substitution is purely "
                "algebraic. Consider using an algebraic solver. Aborting conversion "
                "to semi-explicit form."
            )
            raise AcausalModelError(message=message, dpd=self.dpd)
        eqs = self.final_dae_eqs
        x = self.final_dae_x
        x_dot = self.final_dae_x_dot
        y = self.final_dae_y

        x_dot_y = x_dot.copy()
        x_dot_y.extend(y)

        X = x.copy()
        X.extend(x_dot)
        X.extend(y)

        vars_in_eqs = []
        alg_eqs = []
        diff_eqs = []
        diff_eqs_indices = []
        for eq in eqs:
            d_vars, a_vars = extract_vars(eq, self.known_vars)
            da_vars = d_vars.union(a_vars)
            vars_in_eqs.append(da_vars.difference(x))
            if len(d_vars) == 0:
                alg_eqs.append(eq)
            else:
                diff_eqs.append(eq)
                diff_eqs_indices.append(eqs.index(eq))

        # Create a bipartite graph
        G = nx.Graph()

        # Add nodes with the bipartite attribute. Equation nodes are bipartite 0, and
        # variable nodes are bipartite 1.
        G.add_nodes_from([i for i, _ in enumerate(eqs)], bipartite=0)
        G.add_nodes_from(x_dot_y, bipartite=1)

        # Add edges based on variable presence in each equation
        for eq_idx, vars_in_eq in enumerate(vars_in_eqs):
            for var in vars_in_eq:
                G.add_edge(eq_idx, var)

        v_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

        # Find which equations to use for substibution by maximum matching
        _mm = bipartite.matching.maximum_matching(G, top_nodes=v_nodes)
        mm = {k: v for k, v in _mm.items() if v in v_nodes}
        rmm = {v: k for k, v in mm.items()}

        # Symbolically solve the matched equations for the derivatives
        diff_eqs_idx_to_solve = [
            eq for var, eq in rmm.items() if isinstance(var, sp.Derivative)
        ]
        diff_eqs_to_solve = [eqs[idx] for idx in diff_eqs_idx_to_solve]
        try:
            # if multiple solutions exist, pick the first one [0]
            sol = sp.solve(diff_eqs_to_solve, x_dot, dict=True)[0]
        except Exception as e:
            print(
                "Failed to solve for derivatives explicitly for conversion to "
                "semi-explicit form. Aborting!"
            )
            raise e

        # Create the semi-explicit form
        self.se_x_dot = [var for var in sol.keys()]
        self.se_x_dot_rhs = [var for var in sol.values()]

        self.se_x = [var.expr for var in self.se_x_dot]
        self.se_y = y

        diff_eqs_idx_to_substitute = [
            idx for idx in diff_eqs_indices if idx not in diff_eqs_idx_to_solve
        ]
        diff_eqs_to_substitute = [eqs[idx] for idx in diff_eqs_idx_to_substitute]
        diff_eqs_post_substitution = [eq.subs(sol) for eq in diff_eqs_to_substitute]

        self.se_alg_eqs = alg_eqs.copy()
        self.se_alg_eqs.extend(diff_eqs_post_substitution)

        self.se_dae_X_to_X_mapping = self.dae_X_to_X_mapping.copy()

        if self.ic_computed:
            self.se_x_ic = [
                self.X_ic_mapping[self.dae_X_to_X_mapping[var]] for var in self.se_x
            ]
            self.se_x_dot_ic = [
                self.X_ic_mapping[self.dae_X_to_X_mapping[var]] for var in self.se_x_dot
            ]
            self.se_y_ic = [
                self.X_ic_mapping[self.dae_X_to_X_mapping[var]] for var in self.se_y
            ]

        if self.verbose:
            print("\n")
            print("#" * 10, "Semi-explicit conversion successful", "#" * 10, "\n")
            print(
                "#" * 2,
                "Final DAE equations x_dot = f(x,y,t) & g(x,y)=0",
                "#" * 2,
                "\n",
            )

            print("# with f(x,y,t)=\n")
            for idx, eq in enumerate(self.se_x_dot_rhs):
                print(f"Eq {idx:<4}:  ", eq)

            ndiff = len(self.x_dot)
            print("\n#and g(x,y,t)=\n")

            for idx, eq in enumerate(self.se_alg_eqs):
                print(f"Eq {ndiff + idx:<4}:  ", eq)

            if self.ic_computed:
                print("\n# with, x =\n")
                for var, ic in zip(self.se_x, self.se_x_ic):
                    print(f"{str(var):30} with ic= {ic}")

                print("\n# x_dot =\n")
                for var, ic in zip(self.se_x_dot, self.se_x_dot_ic):
                    print(f"{str(var):30} with ic= {ic}")

                print("\n# and, y =\n")
                for var, ic in zip(self.se_y, self.se_y_ic):
                    print(f"{str(var):30} with ic= {ic}")

                print("\n", "#" * 60, "\n")
            else:
                print("\n# with, x =\n")
                for var in self.se_x:
                    print(f"{str(var)}")

                print("\n# x_dot =\n")
                for var in self.se_x_dot:
                    print(f"{str(var)}")

                print("\n# and, y =\n")
                for var in self.se_y:
                    print(f"{str(var)}")

                print("\n", "#" * 60, "\n")

    def scale_se_equations(self, X_nom_mapping={}, X_ic_mapping={}):
        self.se_x_ic = [
            X_ic_mapping[self.se_dae_X_to_X_mapping[var]] for var in self.se_x
        ]
        self.se_x_dot_ic = [
            X_ic_mapping[self.se_dae_X_to_X_mapping[var]] for var in self.se_x_dot
        ]
        self.se_y_ic = [
            X_ic_mapping[self.se_dae_X_to_X_mapping[var]] for var in self.se_y
        ]

        n_ode = len(self.se_x)
        n_alg = len(self.se_y)

        knowns_symbols, knowns_vals = zip(*self.knowns.items())

        s = self.se_x + self.se_y
        sdot = self.se_x_dot + [sp.Derivative(var, self.t) for var in self.se_y]

        # convert semi-explicit form to implicit representation F(x, x_dot, y, t) = 0
        # for jacobian computation.
        sym_F = [
            xdot - fx for xdot, fx in zip(self.se_x_dot, self.se_x_dot_rhs)
        ] + self.se_alg_eqs

        lambda_args = (self.t, s, sdot, knowns_symbols)

        # ics
        s0 = self.se_x_ic + self.se_y_ic
        sdot0 = self.se_x_dot_ic + [0.0] * n_alg
        s0_map = {var: ic for var, ic in zip(s, s0)}
        sdot0_map = {var: ic for var, ic in zip(sdot, sdot0)}

        # provided nominal values
        X_nom_map = {
            self.se_dae_X_to_X_mapping[var]: ic for var, ic in X_nom_mapping.items()
        }

        s_nom = [
            X_nom_map[var] if var in X_nom_map.keys() else s0_map[var] for var in s
        ]
        sdot_nom = [
            X_nom_map[var] if var in X_nom_map.keys() else sdot0_map[var]
            for var in sdot
        ]

        use_jacobian_scaling = False
        if use_jacobian_scaling:
            # Evaluate the Jacobians at nominal values
            t_nom = 0.0
            # TODO: Symbolic computation of Jacobian fails at the lambidification stage
            # fails when functions such as `Piecewise` and `Abs` are used. We need to
            # look at all the places where Jacobians are symbolically computed and then
            # (i) either make the symbolic computation robust; or (ii) use numerical
            # computation everywhere possible.
            compute_jacobians_symbolically = False
            if compute_jacobians_symbolically:
                sym_dF_ds = sp.Matrix(sym_F).jacobian(s)
                sym_dF_dsdot = sp.Matrix(sym_F).jacobian(sdot)

                dF_ds = sp.lambdify(
                    lambda_args,
                    sym_dF_ds,
                    modules=["jax", {"cnp": cnp}],
                )

                dF_dsdot = sp.lambdify(
                    lambda_args,
                    sym_dF_dsdot,
                    modules=["jax", {"cnp": cnp}],
                )
                J_s = dF_ds(t_nom, s_nom, sdot_nom, knowns_vals)
                J_sdot = dF_dsdot(t_nom, s_nom, sdot_nom, knowns_vals)
            else:
                F = sp.lambdify(
                    lambda_args,
                    sym_F,
                    modules=["jax", {"cnp": cnp}],
                )
                J_s = jax.jacfwd(F, argnums=1)(t_nom, s_nom, sdot_nom, knowns_vals)
                J_sdot = jax.jacfwd(F, argnums=2)(t_nom, s_nom, sdot_nom, knowns_vals)

            # Scaling for the variables x_scaled = x / Ss
            Ss = 1.0 / cnp.max(cnp.abs(cnp.vstack((J_s, J_sdot))), axis=0)
            Ss = cnp.ones(len(s))

            # Scaling for equations eq_scaled = eq / Se
            Se = 1.0 / cnp.max(cnp.abs(cnp.hstack((J_s, J_sdot))), axis=1)
            # Se = cnp.ones(len(sym_F))
        else:
            EPS = 1e-06
            Ss = cnp.array(
                [
                    cnp.abs(var_nom) if cnp.abs(var_nom) >= EPS else 1.0
                    for var_nom in s_nom
                ]
            )
            Se = cnp.ones(len(sym_F))

        var_to_scaled_var = {var: sp.Function(f"scaled_{var}")(self.t) for var in s}

        # scaled variables
        self.ses_x = [var_to_scaled_var[var] for var in self.se_x]
        self.ses_x_dot = [sp.diff(var, self.t) for var in self.ses_x]
        self.ses_y = [var_to_scaled_var[var] for var in self.se_y]

        # Initial condition scaling
        self.ses_x_ic = [val / Ss[idx] for idx, val in enumerate(self.se_x_ic)]
        self.ses_x_dot_ic = [val / Ss[idx] for idx, val in enumerate(self.se_x_dot_ic)]
        self.ses_y_ic = [val / Ss[idx + n_ode] for idx, val in enumerate(self.se_y_ic)]

        # equations with scaled variables
        subs_var_to_scaled_var = {
            var: sp.Function(f"scaled_{var}")(self.t) * scale
            for var, scale in zip(s, Ss)
        }
        self.ses_f = [
            expr.subs(subs_var_to_scaled_var) / Ss[idx]
            for idx, expr in enumerate(self.se_x_dot_rhs)
        ]
        self.ses_g = [expr.subs(subs_var_to_scaled_var) for expr in self.se_alg_eqs]

        # scale equations
        self.ses_g = [expr * Se[idx + n_ode] for idx, expr in enumerate(self.ses_g)]

        self.Ss = Ss

        all_vars_to_scaled_vars = {
            der: scaled_der for der, scaled_der in zip(self.se_x_dot, self.ses_x_dot)
        }
        all_vars_to_scaled_vars.update(var_to_scaled_var)

        self.ses_dae_X_to_X_mapping = {
            all_vars_to_scaled_vars[dae_var]: var
            for dae_var, var in self.se_dae_X_to_X_mapping.items()
        }

        # Create a single dict to be used for substitutions in ZC and output expressions
        self.ses_vars_to_scaled_vars = {
            der: scaled_der * Ss[idx]
            for idx, (der, scaled_der) in enumerate(zip(self.se_x_dot, self.ses_x_dot))
        }
        self.ses_vars_to_scaled_vars.update(subs_var_to_scaled_var)

        self.ses_scaled_vars_to_vars = {
            scaled_der: der / Ss[idx]
            for idx, (der, scaled_der) in enumerate(zip(self.se_x_dot, self.ses_x_dot))
        }
        self.ses_scaled_vars_to_vars.update(
            {
                scaled_var: var / scale
                for scale, var, scaled_var in zip(Ss, s, self.ses_x + self.ses_y)
            }
        )

        if self.verbose:
            print(
                "\n",
                "#" * 10,
                "Scaled equations: ẋ = f(t,x,y) and g(t,x,y)=0",
                "#" * 10,
            )

            print("# with f(x,y,t)=\n")
            for idx, eq in enumerate(self.ses_f):
                print(f"Eq {idx:<4}:  ", eq)

            print("\n#and g(x,y,t)=\n")

            for idx, eq in enumerate(self.ses_g):
                print(f"Eq {n_ode + idx:<4}:  ", eq)

            print("\n# with, x =\n")
            for var, ic in zip(self.ses_x, self.ses_x_ic):
                print(f"{str(var):30} with ic= {ic}")

            print("\n# x_dot =\n")
            for var, ic in zip(self.ses_x_dot, self.ses_x_dot_ic):
                print(f"{str(var):30} with ic= {ic}")

            print("\n# and, y =\n")
            for var, ic in zip(self.ses_y, self.ses_y_ic):
                print(f"{str(var):30} with ic= {ic}")
