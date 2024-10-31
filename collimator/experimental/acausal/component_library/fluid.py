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
from collimator.lazy_loader import LazyLoader

from .base import SymKind, EqnKind
from .component_base import ComponentBase

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")
"""
The concepts are heavily based on these sources:
Modelica Fluids library
https://doc.modelica.org/Modelica%204.0.0/Resources/helpDymola/Modelica_Fluid.html#Modelica.Fluid
The original paper describing the modeling approach. Referenced in 'Modelica Library>Fluid>UsersGuide>ComponentDefinition':
https://modelica.org/events/Conference2003/papers/h40_Elmqvist_fluid.pdf

Modelica Stream connectors concepts:
https://doc.modelica.org/Modelica%204.0.0/Resources/Documentation/Fluid/Stream-Connectors-Overview-Rationale.pdf
https://web.mit.edu/crlaugh/Public/stream-docs.pdf

after searching the MLS code for "Stream", i found this link below in Fluid>package.mo:
https://specification.modelica.org/v3.4/Ch15.html#stream-operator-actualstream

Derivation of stream connector equations:
https://specification.modelica.org/v3.4/A4.html

This implementation omits the stream connector/variables presently.

Defining the properties of the fluid in the components is achieved using the FluidProperties component.
FluidProperties has one acausal port. this port is connected anywhere in the network of components that
all share the same fluid. In this way, it is possible to to have a single AcausalDiagram, but with several
fluid networks, each having different fluid properties. The DiagramProcessing class handles appropriately
assigning the fluid properties from the FluidProperties component to the ports of the components connected
to the network.
"""

M_FLOW_EPS = 1e-8  # HACK: we'll formalize this later
P_DEFAULT = 101325.0
T_DEFAULT = 300.0


class FluidOnePort(ComponentBase):
    """Partial component class for an fluid component with only
    one port.
    """

    def __init__(self, ev, name, P_ic=None, P_ic_fixed=False, p="port"):
        super().__init__()
        self.declare_fluid_port(ev, p, P_ic=P_ic, P_ic_fixed=P_ic_fixed)
        self.port_idx_to_name = {-1: p}
        self.declare_fluid_port_set(set(p))
        self.port_name = p


class FluidTwoPort(ComponentBase):
    """Partial component class for an fluid component with
    two ports.
    """

    def __init__(self, ev, name, p1="port_a", p2="port_b"):
        super().__init__()
        self.declare_fluid_port(ev, p1)
        self.declare_fluid_port(ev, p2)
        self.dP = self.declare_symbol(ev, "dP", name, kind=SymKind.var)
        P1 = self.ports[p1].p
        P2 = self.ports[p2].p
        self.add_eqs([sp.Eq(self.dP.s, P1.s - P2.s)])
        self.port_idx_to_name = {-1: p1, 1: p2}
        self.declare_fluid_port_set(set([p1, p2]))
        self.port_1_name = p1
        self.port_2_name = p2


class FluidProperties(ComponentBase):
    """Component for assigning a Fluid to a set of components by connecting
    the FluidProperties component to the network via its 'prop' port.
    """

    def __init__(self, ev, fluid, name="FluidProperties"):
        super().__init__()
        self.name = name
        # this may not be strictly necessary, but defining the port like this
        # means that all ports of all components conform to the same data, so
        # the same validation check can be used everywhere.
        self.declare_fluid_port(ev, "prop")
        self.fluid = fluid
        self.port_idx_to_name = {-1: "prop"}


class MassflowSource(FluidOnePort):
    """
    Ideal massflow source in the incompressible fluid domain.
    """

    def __init__(self, ev, name=None, M=1.0, enable_massflow_port=False, h_ambient=0.0):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        self.h_ambient = self.declare_symbol(
            ev, "h_ambient", self.name, kind=SymKind.param, val=h_ambient
        )
        self.mflow = self.declare_symbol(
            ev, "mflow", self.name, kind=SymKind.param, val=M
        )

        port = self.ports[self.port_name]
        self.add_eqs([sp.Eq(port.mflow.s, -self.mflow.s)])

    def finalize(self, ev):
        port = self.ports[self.port_name]
        h_outflow = port.stream["h"]
        self.add_eqs([sp.Eq(self.h_ambient.s, h_outflow.s)])
        # add fluid equations
        T = port.T
        u = port.u
        d = port.d
        fluid = port.fluid
        fluid_eqs = fluid.gen_eqs(ev, port.p.s, h_outflow.s, T.s, u.s, d.s)
        self.add_eqs(fluid_eqs, kind=EqnKind.fluid)


class StaticPipe(FluidTwoPort):
    """
    Modelica>Fluid>Pipes>StaticPipe
    """

    def __init__(
        self,
        ev,
        name=None,
        L=1.0,
        D=0.5,
        e=0.0,  # roughness
        h_ab=0.0,
        allow_flow_reversal=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)

        L = self.declare_symbol(
            ev,
            "L",
            self.name,
            kind=SymKind.param,
            val=L,
            validator=lambda x: x > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have L>0",
        )

        D = self.declare_symbol(
            ev,
            "D",
            self.name,
            kind=SymKind.param,
            val=D,
            validator=lambda x: x > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have D>0",
        )

        e = self.declare_symbol(
            ev,
            "e",
            self.name,
            kind=SymKind.param,
            val=e,
            validator=lambda x: x > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have e>0",
        )

        h_ab = self.declare_symbol(
            ev,
            "h_ab",
            self.name,
            kind=SymKind.param,
            val=h_ab,
            validator=lambda x: L >= h_ab,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must L>=h_ab",
        )

        port = self.ports["port_a"]
        fluid = port.fluid
        rho = fluid.density.s
        mu = fluid.viscosity_dyn

        Re = self.declare_symbol(ev, "Re", self.name, kind=SymKind.var)
        f = self.get_friction_factor(Re, e)

        self.add_eqs(
            [
                # does not store mass
                sp.Eq(0, self.M1.s + self.M2.s),
                # Reynolds number
                sp.Eq(
                    Re.s,
                    4 * abs(self.M1.s) / (sp.pi * D.s * mu),
                ),
                # pressure drop equation. use M1 due to sign convention
                sp.Eq(self.dP.s, f * L.s * rho * abs(self.M1.s) ** 2 / (2 * D.s)),
            ]
        )

        def get_friction_factor(Re, e):
            return 64.0 / Re  # for Re<2000


class ClosedVolume(FluidOnePort):
    """see Modelica>Fluid>Vessels>ClosedVolume...sort of."""

    def __init__(
        self,
        ev,
        name=None,
        volume=1.0,
        pressure_ic=P_DEFAULT,
        pressure_ic_fixed=True,
        temperature_ic=T_DEFAULT,
        temperature_ic_fixed=True,
        enable_enthalpy_sensor=False,
        enable_thermal_port=False,
        ht_coeff=0.01,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name, P_ic=pressure_ic, P_ic_fixed=pressure_ic_fixed)

        self.pressure_ic = pressure_ic
        self.pressure_ic_fixed = pressure_ic_fixed
        self.temperature_ic = temperature_ic
        self.temperature_ic_fixed = temperature_ic_fixed
        self.enable_thermal_port = enable_thermal_port

        self.V = self.declare_symbol(ev, "V", self.name, kind=SymKind.param, val=volume)
        self.m = self.declare_symbol(ev, "m", self.name, kind=SymKind.var)
        self.U = self.declare_symbol(ev, "U", self.name, kind=SymKind.var)
        self.dm = self.declare_symbol(
            ev, "dm", self.name, kind=SymKind.var, int_sym=self.m, ic=0.0
        )
        self.m.der_sym = self.dm
        self.dU = self.declare_symbol(
            ev, "dU", self.name, kind=SymKind.var, int_sym=self.U, ic=0.0
        )
        self.U.der_sym = self.dU

        if enable_thermal_port:
            # declare heat port for thermal exchange with other components
            self.thermal_port_name = "wall"
            self.T_wall, self.Q_wall = self.declare_thermal_port(
                ev, self.thermal_port_name, T_ic=T_DEFAULT
            )
            # heat transfer params
            self.ht_coeff = self.declare_symbol(
                ev,
                "ht_coeff",
                self.name,
                kind=SymKind.param,
                val=ht_coeff,
                validator=lambda ht_coeff: ht_coeff > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have ht_coeff>0",
            )

        if enable_enthalpy_sensor:
            self.h_output = self.declare_symbol(
                ev, "h_output", self.name, kind=SymKind.outp
            )
            h_outflow = self.ports[self.port_name].h_outflow
            self.declare_equation(
                sp.Eq(h_outflow.s, self.h_output.s), kind=EqnKind.outp
            )

    def finalize(self, ev):
        port = self.ports[self.port_name]
        if port.fluid is None:
            raise ValueError(
                f"ClosedVolume {self.name} has no fluid assigned to its port."
            )
        h_outflow = port.h_outflow
        h_inStream = port.h_inStream
        T = port.T
        u = port.u
        d = port.d

        fluid_eqs = port.fluid.gen_eqs(ev, port.p.s, h_outflow.s, T.s, u.s, d.s)
        self.add_eqs(fluid_eqs, kind=EqnKind.fluid)

        mflow_in_expr = sp.Piecewise(
            (port.mflow.s, port.mflow.s > M_FLOW_EPS), (M_FLOW_EPS, True)
        )
        mflow_out_expr = sp.Piecewise(
            (port.mflow.s, port.mflow.s < -M_FLOW_EPS), (-M_FLOW_EPS, True)
        )

        # common equations
        self.add_eqs(
            [
                # volume-density-mass constraint
                sp.Eq(self.m.s, self.V.s * d.s),
                # internal energy equation
                sp.Eq(self.U.s, self.m.s * u.s),
                # conservation of mass
                sp.Eq(self.dm.s, port.mflow.s),
            ]
        )
        if not self.enable_thermal_port:
            self.add_eqs(
                [
                    # energy balance
                    sp.Eq(
                        self.dU.s,
                        mflow_in_expr * h_inStream.s + mflow_out_expr * h_outflow.s,
                    )
                ]
            )
        else:
            self.add_eqs(
                [
                    # energy balance
                    sp.Eq(
                        self.dU.s,
                        mflow_in_expr * h_inStream.s
                        + mflow_out_expr * h_outflow.s
                        + self.Q_wall.s,
                    ),
                    sp.Eq(self.Q_wall.s, self.ht_coeff.s * (self.T_wall.s - T.s)),
                ]
            )

        # enforce user assigned ICs
        port.p.ic = self.pressure_ic
        port.p.ic_fixed = self.pressure_ic_fixed
        port.T.ic = self.temperature_ic
        port.T.ic_fixed = self.temperature_ic_fixed
        # compute ICs resulting from these
        h_outflow.ic, u.ic, d.ic = port.fluid.get_h_u_d_ics(
            self.pressure_ic, self.temperature_ic
        )
        self.m.ic = self.V.val * d.ic
        self.U.ic = self.m.ic * u.ic


class Boundary_pT(FluidOnePort):
    """This scould be 'PTSource', sort of like 'PressureSource'."""

    def __init__(
        self,
        ev,
        name=None,
        p_ambient=P_DEFAULT,
        T_ambient=T_DEFAULT,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name, P_ic=p_ambient, P_ic_fixed=True)

        self.p_ambient = self.declare_symbol(
            ev, "p_ambient", self.name, kind=SymKind.param, val=p_ambient
        )
        self.T_ambient = self.declare_symbol(
            ev, "T_ambient", self.name, kind=SymKind.param, val=T_ambient
        )

        port = self.ports[self.port_name]
        self.add_eqs(
            [
                sp.Eq(port.p.s, self.p_ambient.s),
                sp.Eq(port.T.s, self.T_ambient.s),
            ]
        )

    def finalize(self, ev):
        port = self.ports[self.port_name]
        if port.fluid is None:
            raise ValueError(
                f"Boundary_pT {self.name} has no fluid assigned to its port."
            )
        fluid = port.fluid
        # do not need to add equations, because all fluid base properties are
        # constant, so just add equality constraints for the port variables.
        # NOTE: these constraints will eventually be removed by alias elimination.

        # compute boundary fluid props from given props
        h_amb, u_amb, d_amb = fluid.get_h_u_d_ics(
            self.p_ambient.val, self.T_ambient.val
        )

        self.h_ambient = self.declare_symbol(
            ev, "h_ambient", self.name, kind=SymKind.param, val=h_amb
        )
        self.u_ambient = self.declare_symbol(
            ev, "u_ambient", self.name, kind=SymKind.param, val=u_amb
        )
        self.d_ambient = self.declare_symbol(
            ev, "d_ambient", self.name, kind=SymKind.param, val=d_amb
        )
        self.add_eqs(
            [
                sp.Eq(port.h_outflow.s, self.h_ambient.s),
                sp.Eq(port.u.s, self.u_ambient.s),
                sp.Eq(port.d.s, self.d_ambient.s),
            ]
        )


class SimplePipe(FluidTwoPort):
    """
    Simple pipe in the incompressible fluid domain.The characteristic equation is:
    P1(t) - P2(t) = massflow(t)*R.

    This is obviously overly simplified, but an acceptable starting point.
    """

    def __init__(self, ev, name=None, R=1.0, enable_sensors=False):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        self.enable_sensors = enable_sensors
        self.R = self.declare_symbol(
            ev,
            "R",
            self.name,
            kind=SymKind.param,
            val=R,
            validator=lambda R: R > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have R>0",
        )

        if self.enable_sensors:
            self.m_flow = self.declare_symbol(
                ev, "m_flow", self.name, kind=SymKind.outp
            )
            self.pa = self.declare_symbol(ev, "pa", self.name, kind=SymKind.outp)
            self.pb = self.declare_symbol(ev, "pb", self.name, kind=SymKind.outp)

    def finalize(self, ev):
        def add_fld_eqs(p):
            # self.add_eqs(
            #     p.fluid.gen_eqs(ev, p.p.s, p.h_outflow.s, p.T.s, p.u.s, p.d.s),
            #     kind=EqnKind.fluid,
            # )
            return p.h_outflow.s, p.h_inStream.s

        p1 = self.ports[self.port_1_name]
        p2 = self.ports[self.port_2_name]
        if p1.fluid is None or p2.fluid is None:
            raise ValueError(
                f"SimplePipe {self.name} has no fluid assigned to its port."
            )

        h_out1, h_in1 = add_fld_eqs(p1)
        h_out2, h_in2 = add_fld_eqs(p2)
        self.add_eqs(
            [
                # does not store mass
                sp.Eq(0, p1.mflow.s + p2.mflow.s),
                # pressure drop equation. use M1 due to sign convention
                sp.Eq(self.dP.s, self.R.s * p1.mflow.s),
                # basic energy conservation
                sp.Eq(h_out2, h_in1),
                sp.Eq(h_out1, h_in2),
            ]
        )
        if self.enable_sensors:
            self.declare_equation(sp.Eq(p1.mflow.s, self.m_flow.s), kind=EqnKind.outp)
            self.declare_equation(sp.Eq(p1.p.s, self.pa.s), kind=EqnKind.outp)
            self.declare_equation(sp.Eq(p2.p.s, self.pb.s), kind=EqnKind.outp)

        p1.p.ic = p1.fluid.init["p"]
        p1.T.ic = p1.fluid.init["T"]
        p1.h_outflow.ic = p1.fluid.init["h"]
        p1.u.ic = p1.fluid.init["u"]
        p1.d.ic = p1.fluid.init["d"]

        p2.p.ic = p2.fluid.init["p"]
        p2.T.ic = p2.fluid.init["T"]
        p2.h_outflow.ic = p2.fluid.init["h"]
        p2.u.ic = p2.fluid.init["u"]
        p2.d.ic = p2.fluid.init["d"]

        # NOTE: we dont have ICs for pressure, temperature, nor enthalpy, at either end of the pipe.
        # therefore so we cant reasonably compute ICs for for other fluid properties either.
        # it is not reasonable to expect a user to provide ICs for even pressure at all pipe ends in a model, as this
        # is sort of what they are using the model to solve for them.


class PTSensor(ComponentBase):
    """
    Ideal pressure and temperature sensor in the fluid domain.
    When port_b enabled, measures between port_a and port_b.
    When port_b disabled, measures the absolute pressure (Pa) and temperature (K).

    NOTE: presently it is a Pressure+Enthalpy sensor, until we can get the solver hanging issue sorted.
    """

    def __init__(
        self,
        ev,
        name=None,
        enable_port_b=True,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__()
        self.enable_port_b = enable_port_b
        self.port_1_name = "port_a"
        self.declare_fluid_port(ev, self.port_1_name, no_mflow=True)
        self.port_idx_to_name = {-1: self.port_1_name}
        fluid_port_set = [self.port_1_name]
        if enable_port_b:
            self.port_2_name = "port_b"
            self.declare_fluid_port(ev, self.port_2_name, no_mflow=True)
            self.port_idx_to_name[1] = self.port_2_name
            fluid_port_set.append(self.port_2_name)
        self.declare_fluid_port_set(fluid_port_set)

        p = self.declare_symbol(ev, "p", self.name, kind=SymKind.outp)
        temp = self.declare_symbol(ev, "temp", self.name, kind=SymKind.outp)
        p1 = self.ports[self.port_1_name].p
        # T1 = self.ports[self.port_1_name].T
        h_inStream1 = self.ports[self.port_1_name].h_inStream
        mflow1 = self.ports[self.port_1_name].mflow
        if enable_port_b:
            p2 = self.ports[self.port_2_name].p
            # T2 = self.ports[self.port_2_name].T
            mflow2 = self.ports[self.port_2_name].mflow
            self.declare_equation(sp.Eq(p.s, p1.s - p2.s), kind=EqnKind.outp)
            # self.declare_equation(sp.Eq(temp.s, T1.s - T2.s), kind=EqnKind.outp)
            self.add_eqs([sp.Eq(mflow1.s, 0), sp.Eq(mflow2.s, 0)])
        else:
            self.declare_equation(sp.Eq(p.s, p1.s), kind=EqnKind.outp)
            # self.declare_equation(sp.Eq(temp.s, T1.s), kind=EqnKind.outp)
            self.declare_equation(sp.Eq(temp.s, h_inStream1.s), kind=EqnKind.outp)
            self.add_eqs([sp.Eq(mflow1.s, 0)])

    # FIXME: including the fluid equations for the sensor causes the collimator solver to hang.
    # def finalize(self, ev):
    #     def add_fld_eqs(p):
    #         self.add_eqs(
    #             p.fluid.gen_eqs(ev, p.p.s, p.h_inStream.s, p.T.s, p.u.s, p.d.s),
    #             kind=EqnKind.fluid,
    #         )

    #     p1 = self.ports[self.port_1_name]
    #     if p1.fluid is None:
    #         raise ValueError(
    #             f"PTSensor {self.name} has no fluid assigned to its port(s)."
    #         )
    #     add_fld_eqs(p1)

    #     if self.enable_port_b:
    #         p2 = self.ports[self.port_2_name]
    #         if p2.fluid is None:
    #             raise ValueError(
    #                 f"PTSensor {self.name} has no fluid assigned to its port(s)."
    #             )
    #         add_fld_eqs(p2)


class MassflowSensor(FluidTwoPort):
    """.
    Ideal massflow sensor in the fluid domain.
    """

    def __init__(
        self,
        ev,
        name=None,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        self.m_flow = self.declare_symbol(ev, "m_flow", self.name, kind=SymKind.outp)
        # self.declare_equation(sp.Eq(m_flow.s, self.M1.s), kind=EqnKind.outp)
        # self.add_eqs(
        #     [
        #         # does not accumulate mass
        #         sp.Eq(0, self.M1.s + self.M2.s),
        #         # no pressure drop
        #         sp.Eq(self.P1.s, self.P2.s),
        #     ]
        # )

    def finalize(self, ev):
        def add_fld_eqs(p):
            # self.add_eqs(
            #     p.fluid.gen_eqs(ev, p.p.s, p.h_outflow.s, p.T.s, p.u.s, p.d.s),
            #     kind=EqnKind.fluid,
            # )
            return p.h_outflow.s, p.h_inStream.s

        p1 = self.ports[self.port_1_name]
        p2 = self.ports[self.port_2_name]
        if p1.fluid is None or p1.fluid is None:
            raise ValueError(
                f"SimplePipe {self.name} has no fluid assigned to its port."
            )

        h_out1, h_in1 = add_fld_eqs(p1)
        h_out2, h_in2 = add_fld_eqs(p2)
        self.add_eqs(
            [
                # does not store mass
                sp.Eq(0, p1.mflow.s + p2.mflow.s),
                # pressure drop equation. not pressure loss in ideal massflow sensor
                sp.Eq(self.dP.s, 0),
                # basic energy conservation
                sp.Eq(h_out2, h_in1),
                sp.Eq(h_out1, h_in2),
            ]
        )
        self.declare_equation(sp.Eq(p1.mflow.s, self.m_flow.s), kind=EqnKind.outp)

        p1.p.ic = p1.fluid.init["p"]
        p1.T.ic = p1.fluid.init["T"]
        p1.h_outflow.ic = p1.fluid.init["h"]
        p1.u.ic = p1.fluid.init["u"]
        p1.d.ic = p1.fluid.init["d"]

        p2.p.ic = p2.fluid.init["p"]
        p2.T.ic = p2.fluid.init["T"]
        p2.h_outflow.ic = p2.fluid.init["h"]
        p2.u.ic = p2.fluid.init["u"]
        p2.d.ic = p2.fluid.init["d"]

        # NOTE: same as pipe


class Accumulator(FluidOnePort):
    """
    Accumulator for incompressible fluid. Pressure increases when mass flows in, and vice versa.
    The relationship between internal pressure and mass flow is spring law.
    There is no restrictor at the port, no pressure loss a function of flow rate.

    Note: Initial pressure P_ic must be set. It has been observed that trying to use weak ICs for this
    i.e. leaving P_ic as None, produces problematic system equations.

    Spring law equations.
        massflow/(fluid.density) = der(V)
        force = (V/area)*k
        pressure = force/area

        V_init = f*area/k = pressure*area*area/k (N/m*m)*(m*m)*(m*m)/(N/m)->(N)*(m*m)/(N/m)->m**3

    Args:
        P_ic (number):
            Initial pressure of the accumulator.
        area (number):
            The surface area acted on by the accumulator internal pressure.
        k (number):
            Spring stiffness.

    """

    def __init__(
        self,
        ev,
        name=None,
        P_ic=P_DEFAULT,
        P_ic_fixed=True,
        T_ic=T_DEFAULT,
        T_ic_fixed=False,
        area=1.0,
        k=1.0,
        enable_enthalpy_sensor=False,
        enable_thermal_port=False,
        ht_coeff=0.01,
    ):
        self.name = self.__class__.__name__ if name is None else name

        super().__init__(ev, self.name, P_ic=P_ic, P_ic_fixed=P_ic_fixed)
        self.P_ic = P_ic
        self.P_ic_fixed = P_ic_fixed
        self.T_ic = T_ic
        self.T_ic_fixed = T_ic_fixed
        self.V_ic = P_ic * area * area / k
        self.enable_thermal_port = enable_thermal_port
        area = self.declare_symbol(
            ev,
            "area",
            self.name,
            kind=SymKind.param,
            val=area,
            validator=lambda area: area > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have area>0",
        )
        k = self.declare_symbol(
            ev,
            "k",
            self.name,
            kind=SymKind.param,
            val=k,
            validator=lambda k: k > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have k>0",
        )
        f = self.declare_symbol(ev, "f", self.name, kind=SymKind.var)
        self.V = self.declare_symbol(ev, "V", self.name, kind=SymKind.var, ic=self.V_ic)
        self.dV = self.declare_symbol(
            ev, "dV", self.name, kind=SymKind.var, int_sym=self.V, ic=0.0
        )
        self.V.der_sym = self.dV
        self.m = self.declare_symbol(ev, "m", self.name, kind=SymKind.var)
        self.U = self.declare_symbol(ev, "U", self.name, kind=SymKind.var)
        self.dm = self.declare_symbol(
            ev, "dm", self.name, kind=SymKind.var, int_sym=self.m, ic=0.0
        )
        self.m.der_sym = self.dm
        self.dU = self.declare_symbol(
            ev, "dU", self.name, kind=SymKind.var, int_sym=self.U, ic=0.0
        )
        self.U.der_sym = self.dU

        p = self.ports[self.port_name].p
        self.add_eqs(
            [
                # pressure-force relationship
                sp.Eq(p.s, f.s / area.s),
                # force-volume relationship for 'spring' accumulator.
                sp.Eq(f.s, self.V.s * k.s / area.s),
            ]
        )

        if enable_thermal_port:
            # declare heat port for thermal exchange with other components
            self.thermal_port_name = "wall"
            self.T_wall, self.Q_wall = self.declare_thermal_port(
                ev, self.thermal_port_name, T_ic=T_DEFAULT
            )
            # heat transfer param
            self.ht_coeff = self.declare_symbol(
                ev,
                "ht_coeff",
                self.name,
                kind=SymKind.param,
                val=ht_coeff,
                validator=lambda ht_coeff: ht_coeff > 0.0,
                invalid_msg=f"Component {self.__class__.__name__} {self.name} must have ht_coeff>0",
            )

            # debug heat flow sensor
            sens_Qwall = self.declare_symbol(
                ev, "sens_Qwall", self.name, kind=SymKind.outp
            )
            self.declare_equation(sp.Eq(self.Q_wall.s, sens_Qwall.s), kind=EqnKind.outp)

        if enable_enthalpy_sensor:
            self.h_output = self.declare_symbol(
                ev, "h_output", self.name, kind=SymKind.outp
            )
            h_outflow = self.ports[self.port_name].h_outflow
            self.declare_equation(
                sp.Eq(h_outflow.s, self.h_output.s), kind=EqnKind.outp
            )

    def finalize(self, ev):
        port = self.ports[self.port_name]
        if port.fluid is None:
            raise ValueError(
                f"ClosedVolume {self.name} has no fluid assigned to its port."
            )
        h_outflow = port.h_outflow
        h_inStream = port.h_inStream
        T = port.T
        u = port.u
        d = port.d
        fluid_eqs = port.fluid.gen_eqs(ev, port.p.s, h_outflow.s, T.s, u.s, d.s)
        self.add_eqs(fluid_eqs, kind=EqnKind.fluid)

        mflow_in_expr = sp.Piecewise(
            (port.mflow.s, port.mflow.s > M_FLOW_EPS), (M_FLOW_EPS, True)
        )
        mflow_out_expr = sp.Piecewise(
            (port.mflow.s, port.mflow.s < -M_FLOW_EPS), (-M_FLOW_EPS, True)
        )
        # common equations
        self.add_eqs(
            [
                # volume-density-mass constraint
                sp.Eq(self.m.s, self.V.s * d.s),
                # internal energy equation
                sp.Eq(self.U.s, self.m.s * u.s),
                # conservation of mass
                sp.Eq(self.dm.s, port.mflow.s),
            ]
        )
        if not self.enable_thermal_port:
            self.add_eqs(
                [
                    # energy balance
                    sp.Eq(
                        self.dU.s,
                        mflow_in_expr * h_inStream.s + mflow_out_expr * h_outflow.s,
                    )
                ]
            )
        else:
            self.add_eqs(
                [
                    # energy balance
                    sp.Eq(
                        self.dU.s,
                        mflow_in_expr * h_inStream.s
                        + mflow_out_expr * h_outflow.s
                        + self.Q_wall.s,
                    ),
                    sp.Eq(self.Q_wall.s, self.ht_coeff.s * (self.T_wall.s - T.s)),
                ]
            )

        # enforce user assigned ICs
        port.p.ic = self.P_ic
        port.p.ic_fixed = self.P_ic_fixed
        port.T.ic = self.T_ic
        port.T.ic_fixed = self.T_ic_fixed
        # compute ICs resulting from these
        h_outflow.ic, u.ic, d.ic = port.fluid.get_h_u_d_ics(self.P_ic, self.T_ic)
        self.m.ic = self.V.ic * d.ic
        self.U.ic = self.m.ic * u.ic

        # HACKY addition of a bunch of debugging sensors for variables related to Accumulator
        T_fluid_out = self.declare_symbol(
            ev, "T_fluid_out", self.name, kind=SymKind.outp
        )
        self.declare_equation(sp.Eq(T.s, T_fluid_out.s), kind=EqnKind.outp)

        port_mflow = self.declare_symbol(ev, "port_mflow", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(port.mflow.s, port_mflow.s), kind=EqnKind.outp)

        sens_h_inStream = self.declare_symbol(
            ev, "sens_h_inStream", self.name, kind=SymKind.outp
        )
        self.declare_equation(sp.Eq(h_inStream.s, sens_h_inStream.s), kind=EqnKind.outp)

        sens_mass = self.declare_symbol(ev, "sens_mass", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(self.m.s, sens_mass.s), kind=EqnKind.outp)

        sens_U = self.declare_symbol(ev, "sens_U", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(self.U.s, sens_U.s), kind=EqnKind.outp)

        sens_u = self.declare_symbol(ev, "sens_u", self.name, kind=SymKind.outp)
        self.declare_equation(sp.Eq(u.s, sens_u.s), kind=EqnKind.outp)


class OpenTank(FluidOnePort):
    """
    OpenTank for incompressible fluid. Pressure increases when mass flows in, and vice versa.
    The relationship between internal pressure and mass flow is related to gravity and ambient
    pressure condition.
    There is no restrictor at the port, no pressure loss a function of flow rate.
    The port is located at the bottom of the tank.

    Port pressure relationship:
        Pport = P_amb - tank_fluid_height*fluid_density

    Args:
        P_amb (number):
            Ambiant pressure acting on the tank.
        area (number):
            The cross sectional area of the tank.
        H_ic (number):
            Initial height of fluid in the tank.

    """

    def __init__(
        self,
        ev,
        name=None,
        P_amb=P_DEFAULT,
        T_ic=T_DEFAULT,
        T_ic_fixed=False,
        area=1.0,
        h_ic=1.0,  # cannot be zero
        enabble_h_sensor=False,
    ):
        self.name = self.__class__.__name__ if name is None else name

        super().__init__(ev, self.name, P_ic=P_amb, P_ic_fixed=True)
        self.P_amb = P_amb
        self.T_ic = T_ic
        self.T_ic_fixed = T_ic_fixed
        self.V_ic = h_ic * area
        self.area = self.declare_symbol(
            ev,
            "area",
            self.name,
            kind=SymKind.param,
            val=area,
            validator=lambda area: area > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have area>0",
        )
        self.P_amb = self.declare_symbol(
            ev,
            "p_amb",
            self.name,
            kind=SymKind.param,
            val=P_amb,
            validator=lambda area: area > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have P_amb>0",
        )
        self.h = self.declare_symbol(
            ev,
            "h",
            self.name,
            kind=SymKind.var,
            ic=h_ic,
        )
        self.V = self.declare_symbol(ev, "V", self.name, kind=SymKind.var, ic=self.V_ic)
        self.dV = self.declare_symbol(
            ev, "dV", self.name, kind=SymKind.var, int_sym=self.V, ic=0.0
        )
        self.V.der_sym = self.dV
        self.m = self.declare_symbol(ev, "m", self.name, kind=SymKind.var)
        self.U = self.declare_symbol(ev, "U", self.name, kind=SymKind.var)
        self.dm = self.declare_symbol(
            ev, "dm", self.name, kind=SymKind.var, int_sym=self.m, ic=0.0
        )
        self.m.der_sym = self.dm
        self.dU = self.declare_symbol(
            ev, "dU", self.name, kind=SymKind.var, int_sym=self.U, ic=0.0
        )
        self.U.der_sym = self.dU

        self.enabble_h_sensor = enabble_h_sensor
        if enabble_h_sensor:
            self.height_output = self.declare_symbol(
                ev, "height_output", self.name, kind=SymKind.outp
            )
            self.declare_equation(
                sp.Eq(self.h.s, self.height_output.s), kind=EqnKind.outp
            )

    def finalize(self, ev):
        port = self.ports[self.port_name]
        if port.fluid is None:
            raise ValueError(
                f"ClosedVolume {self.name} has no fluid assigned to its port."
            )
        h_outflow = port.h_outflow
        h_inStream = port.h_inStream
        T = port.T
        u = port.u
        d = port.d
        fluid_eqs = port.fluid.gen_eqs(ev, port.p.s, h_outflow.s, T.s, u.s, d.s)
        self.add_eqs(fluid_eqs, kind=EqnKind.fluid)

        mflow_in_expr = sp.Piecewise(
            (port.mflow.s, port.mflow.s > M_FLOW_EPS), (M_FLOW_EPS, True)
        )
        mflow_out_expr = sp.Piecewise(
            (port.mflow.s, port.mflow.s < -M_FLOW_EPS), (-M_FLOW_EPS, True)
        )
        self.add_eqs(
            [
                # volume-height constraint
                sp.Eq(self.V.s, self.h.s * self.area.s),
                # volume-density-mass constraint
                sp.Eq(self.m.s, self.V.s * d.s),
                # internal energy equation
                sp.Eq(self.U.s, self.m.s * u.s),
                # conservation of mass
                sp.Eq(self.dm.s, port.mflow.s),
                # energy balance
                sp.Eq(
                    self.dU.s,
                    mflow_in_expr * h_inStream.s + mflow_out_expr * h_outflow.s,
                ),
                # port pressure constraint
                sp.Eq(port.p.s, self.P_amb.s + ev.g_n.s * self.h.s),
            ]
        )

        # enforce user assigned ICs
        P_ic = self.P_amb.val + ev.g_n.val * self.h.ic
        port.p.ic = P_ic
        port.p.ic_fixed = True
        port.T.ic = self.T_ic
        port.T.ic_fixed = self.T_ic_fixed
        # compute ICs resulting from these
        h_outflow.ic, u.ic, d.ic = port.fluid.get_h_u_d_ics(P_ic, self.T_ic)
        self.m.ic = self.V.ic * d.ic
        self.U.ic = self.m.ic * u.ic


class ThermalPipe(FluidTwoPort):
    """
    Simple pipe didn't work, or I didn't try hard enough. see above.

    Try using a pipe with a defined length and crossectional area, this has a defined volume, and
    for given fluid properties, a defined mass. Having a defined mass allows using 'U' instead of 'u'
    for the energy balance.

    STOP: try thermal port on Accumlator first.
    """

    def __init__(
        self,
        ev,
        name=None,
        R=1.0,  # pressure drop per unit length
        L=1.0,  # length
        A=0.1,  # cross-sectional area
        ht_coeff=1.0,  # heat transfer per per degree K
        enable_sensors=False,
    ):
        self.name = self.__class__.__name__ if name is None else name
        super().__init__(ev, self.name)
        self.enable_sensors = enable_sensors

        # flow model params
        self.R = self.declare_symbol(
            ev,
            "R",
            self.name,
            kind=SymKind.param,
            val=R,
            validator=lambda R: R > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have R>0",
        )

        # sensor configuration
        if self.enable_sensors:
            self.m_flow = self.declare_symbol(
                ev, "m_flow", self.name, kind=SymKind.outp
            )
            self.pa = self.declare_symbol(ev, "pa", self.name, kind=SymKind.outp)
            self.pb = self.declare_symbol(ev, "pb", self.name, kind=SymKind.outp)
            self.sensU = self.declare_symbol(ev, "sensU", self.name, kind=SymKind.outp)
            self.sensQ = self.declare_symbol(ev, "sensQ", self.name, kind=SymKind.outp)
            self.sensHin1 = self.declare_symbol(
                ev, "sensHin1", self.name, kind=SymKind.outp
            )
            self.sensTin1 = self.declare_symbol(
                ev, "sensTin1", self.name, kind=SymKind.outp
            )

        # declare heat port for thermal exchange with other components
        self.thermal_port_name = "wall"
        self.T_wall, self.Q_wall = self.declare_thermal_port(
            ev, self.thermal_port_name, T_ic=T_DEFAULT
        )
        # heat transfer params
        self.ht_coeff = self.declare_symbol(
            ev,
            "ht_coeff",
            self.name,
            kind=SymKind.param,
            val=ht_coeff,
            validator=lambda ht_coeff: ht_coeff > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have ht_coeff>0",
        )
        V_val = A * L
        self.V = self.declare_symbol(
            ev,
            "V",
            self.name,
            kind=SymKind.param,
            val=V_val,
            validator=lambda V_val: V_val > 0.0,
            invalid_msg=f"Component {self.__class__.__name__} {self.name} must have V=L*A>0",
        )
        self.U = self.declare_symbol(ev, "U", self.name, kind=SymKind.var)
        self.dU = self.declare_symbol(
            ev, "dU", self.name, kind=SymKind.var, int_sym=self.U, ic=0.0
        )
        self.U.der_sym = self.dU

    def finalize(self, ev):
        def add_fld_eqs(p):
            self.add_eqs(
                p.fluid.gen_eqs(ev, p.p.s, p.h_outflow.s, p.T.s, p.u.s, p.d.s),
                kind=EqnKind.fluid,
            )
            return p.h_outflow.s, p.h_inStream.s

        p1 = self.ports[self.port_1_name]
        p2 = self.ports[self.port_2_name]
        if p1.fluid is None or p2.fluid is None:
            raise ValueError(
                f"SimplePipe {self.name} has no fluid assigned to its port."
            )

        h_out1, h_inS1 = add_fld_eqs(p1)
        h_out2, h_inS2 = add_fld_eqs(p2)

        # declare vars for the inStream1 properties
        sym_base_name = self.name + "_" + self.port_1_name
        T_inS1 = self.declare_symbol(ev, "T_inS", sym_base_name, kind=SymKind.var)
        u_inS1 = self.declare_symbol(ev, "u_inS", sym_base_name, kind=SymKind.var)
        d_inS1 = self.declare_symbol(ev, "d_inS", sym_base_name, kind=SymKind.var)
        # fluid media equations for inS1
        self.add_eqs(
            p1.fluid.gen_eqs(ev, p1.p.s, h_inS1, T_inS1.s, u_inS1.s, d_inS1.s),
            kind=EqnKind.fluid,
        )

        # declare vars for the inStream1 properties
        # sym_base_name = self.name + "_" + self.port_2_name
        # T_inS2 = self.declare_symbol(ev, "T_inS", sym_base_name, kind=SymKind.var)
        # u_inS2 = self.declare_symbol(ev, "u_inS", sym_base_name, kind=SymKind.var)
        # d_inS2 = self.declare_symbol(ev, "d_inS", sym_base_name, kind=SymKind.var)
        # # fluid media equations for inS2
        # self.add_eqs(
        #     p2.fluid.gen_eqs(ev, p2.p.s, h_inS2, T_inS2.s, u_inS2.s, d_inS2.s),
        #     kind=EqnKind.fluid,
        # )

        # Q12 = self.declare_symbol(ev, "Q12", self.name, kind=SymKind.var)
        # Q21 = self.declare_symbol(ev, "Q21", sym_base_name, kind=SymKind.var)

        mflow1_expr = sp.Piecewise(
            (p1.mflow.s, p1.mflow.s > M_FLOW_EPS), (M_FLOW_EPS, True)
        )
        # mflow2_expr = sp.Piecewise(
        #     (p2.mflow.s, p2.mflow.s > M_FLOW_EPS), (M_FLOW_EPS, True)
        # )

        # Q_expr = sp.Piecewise((Q12.s, p1.mflow.s > M_FLOW_EPS), (Q21.s, True))

        # common equations
        self.add_eqs(
            [
                # does not store mass
                sp.Eq(0, p1.mflow.s + p2.mflow.s),
                # pressure drop equation. use M1 due to sign convention
                sp.Eq(self.dP.s, self.R.s * p1.mflow.s),
                # volume-density-mass and internal energy constraints
                sp.Eq(self.U.s, self.V.s * d_inS1.s * u_inS1.s),
                # energy balance
                sp.Eq(
                    self.dU.s,
                    mflow1_expr * (h_inS1 - h_out2) + self.Q_wall.s,
                ),
                sp.Eq(self.Q_wall.s, self.ht_coeff.s * (self.T_wall.s - T_inS1.s)),
                # HACK: no heat transfer in reverse flow
                sp.Eq(h_out1, h_inS2),
            ]
        )

        if self.enable_sensors:
            self.declare_equation(sp.Eq(p1.mflow.s, self.m_flow.s), kind=EqnKind.outp)
            self.declare_equation(sp.Eq(p1.p.s, self.pa.s), kind=EqnKind.outp)
            self.declare_equation(sp.Eq(p2.p.s, self.pb.s), kind=EqnKind.outp)

            self.declare_equation(sp.Eq(self.U.s, self.sensU.s), kind=EqnKind.outp)
            self.declare_equation(sp.Eq(self.Q_wall.s, self.sensQ.s), kind=EqnKind.outp)
            self.declare_equation(sp.Eq(h_inS1, self.sensHin1.s), kind=EqnKind.outp)
            self.declare_equation(sp.Eq(T_inS1.s, self.sensTin1.s), kind=EqnKind.outp)

        p1.p.ic = p1.fluid.init["p"]
        p1.T.ic = p1.fluid.init["T"]
        p1.h_outflow.ic = p1.fluid.init["h"]
        p1.u.ic = p1.fluid.init["u"]
        p1.d.ic = p1.fluid.init["d"]

        p2.p.ic = p2.fluid.init["p"]
        p2.T.ic = p2.fluid.init["T"]
        p2.h_outflow.ic = p2.fluid.init["h"]
        p2.u.ic = p2.fluid.init["u"]
        p2.d.ic = p2.fluid.init["d"]

        # NOTE: we dont have ICs for pressure, temperature, nor enthalpy, at either end of the pipe.
        # therefore so we cant reasonably compute ICs for for other fluid properties either.
        # it is not reasonable to expect a user to provide ICs for even pressure at all pipe ends in a model, as this
        # is sort of what they are using the model to solve for them.


# anything below here is old, disregard for now.
# class HydraulicActuatorLinear(FluidTwoPort):
#     """
#     Converters pressure to force, and mass_flow to translational motion.
#     Component has 2 mechanical connections like a spring, flange_a and flange_b.
#     Components has 2 fluid connections, like a hydraulic actuator, port_a, and port_b.
#     dP=P1-P2
#     dF=F1-F2
#     Assuming all Ps and Fs are >0, when dP*area>pF, p1 and p2 get further apart, and vice versa.
#     I'm not sure they have to all be positive for that to hold, but I know when they are, it does.
#     """

#     def __init__(self, ev, name=None, area=1.0, x_ic=0.0, x_ic_fixed=False):
#         self.name = self.__class__.__name__ if name is None else name
#         super().__init__(ev, self.name)
#         f1, x1, v1, a1 = self.declare_translational_port(ev, "flange_a")
#         f2, x2, v2, a2 = self.declare_translational_port(
#             ev,
#             "flange_b",
#             x_ic=x_ic,
#             x_ic_fixed=x_ic_fixed,
#         )
#         self.add_eqs([sp.Eq(0, f1.s + f2.s)])

#         V_ic = x_ic * area
#         V = self.declare_symbol(ev, "V", self.name, kind=SymKind.var, ic=V_ic)
#         area = self.declare_symbol(
#             ev,
#             "area",
#             self.name,
#             kind=SymKind.param,
#             val=area,
#             validator=lambda area: area > 0.0,
#             invalid_msg=f"Component {self.__class__.__name__} {self.name} must have area>0",
#         )
#         self.dV = self.declare_symbol(
#             ev, "dV", self.name, kind=SymKind.var, int_sym=V, ic=0.0
#         )
#         V.der_sym = self.dV

#         self.add_eqs(
#             [
#                 # force<->pressure relationships
#                 sp.Eq(self.dP.s * area.s, -f2.s),
#                 # the 'tracked' volume of fluid increases when the actuator gets longer.
#                 sp.Eq(V.s / area.s, x1.s - x2.s),
#                 # mass_flow constrain. e.g. conservation of mass.
#                 # here we consider the actuator 'ideal' in the sense that piston area on both sides is equal.
#                 # in reality this is not true, because there is typically one side with a rod which reduces
#                 # the piston area.
#                 sp.Eq(0, self.M1.s + self.M2.s),
#             ]
#         )

#     def finalize(self):
#         # velocity<->mass_flow relationships
#         # the 'tracked' volume of fluid in the actuator increases when M1 is positive
#         fluid = self.ports["port_a"].fluid
#         self.add_eqs(
#             [
#                 sp.Eq(self.dV.s, self.M1.s / fluid.density.s),
#             ]
#         )


# class PressureSource(FluidOnePort):
#     """
#     Ideal pressure source in the incompressible fluid domain.

#     Args:
#         pressure (number):
#             Pressure value when enable_speed_port=False.
#         enable_pressure_port (bool):
#             When true, the pressure value is from a input signal. When false,
#             pressure value is from 'pressure'.
#     """

#     def __init__(self, ev, name=None, pressure=0.1, enable_pressure_port=False):
#         self.name = self.__class__.__name__ if name is None else name
#         super().__init__(ev, self.name)

#         if enable_pressure_port:
#             kind = SymKind.inp
#             val = None
#         else:
#             kind = SymKind.param
#             val = pressure

#         # in this case we have to create an additional symbol since it is not OK
#         # to change the kind of a potential/flow variable.
#         pressure = self.declare_symbol(ev, "pressure", self.name, kind=kind, val=val)
#         self.add_eqs([sp.Eq(self.P.s, pressure.s)])  # pressure source equality


# class Pump(FluidTwoPort):
#     """
#     Ideal pump in the incompressible fluid domain.

#     The pump has a maximum pressure differential, dP_max, which it can produce.
#     Therefore, the max outlet pressure is dP_max - P_in.
#     The pump has some input power, pwr, which is a causal input signal,
#     that represents the external work done by the pump on the fluid
#     system.
#     The pump has some performance coefficient that converts dP*pwr to massflow,
#     CoP, which has units of kg*s*MPa/Watt.
#         massflow = pwr * CoP / ((dP_max - Pin) - P_out) [1]
#         but dP = P_out - P_in -> P_out = dP + P_in [2]
#     subing [2] into [1] and rearranging:
#         massflow = pwr * CoP / (dP_max - dP)

#     Note on sign convention.
#     Pump component has 2 ports: p1=inlet and p2=outlet.
#     massflow is defined as positive going into the component.
#     Therefore the inlet massflow is positive when pump operating normally.

#     """

#     def __init__(
#         self,
#         ev,
#         name=None,
#         dPmax=1.0,
#         CoP=1.0,
#     ):
#         self.name = self.__class__.__name__ if name is None else name

#         super().__init__(ev, self.name)
#         dPmax = self.declare_symbol(
#             ev,
#             "dPmax",
#             self.name,
#             kind=SymKind.param,
#             val=dPmax,
#             validator=lambda dPmax: dPmax > 0.0,
#             invalid_msg=f"Component {self.__class__.__name__} {self.name} must have dPmax>0",
#         )
#         CoP = self.declare_symbol(
#             ev,
#             "CoP",
#             self.name,
#             kind=SymKind.param,
#             val=CoP,
#             validator=lambda CoP: CoP > 0.0,
#             invalid_msg=f"Component {self.__class__.__name__} {self.name} must have CoP>0",
#         )
#         pwr = self.declare_symbol(ev, "pwr", self.name, kind=SymKind.inp)
#         self.add_eqs(
#             [
#                 # does not store mass
#                 sp.Eq(0, self.M1.s + self.M2.s),
#                 # pressure-massflow relationship.
#                 sp.Eq(self.M1.s, pwr.s * CoP.s / (dPmax.s - self.dP.s)),
#             ]
#         )
