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

import sympy as sp
import numpy as np
from .base import Sym, SymKind


"""
Fluid media specification classes.
based on the media modeling concepts in:
https://modelica.org/events/Conference2003/papers/h40_Elmqvist_fluid.pdf
"""

# some notes from modelica docs.
# "Base properties (p, d, T, h, u, R_s, MM and, if applicable, X and Xi) of a medium"
# SpecificHeatCapacity R_s "Gas constant (of mixture if applicable)";
# MolarMass MM "Molar mass (of mixture or single fluid)";

# 1] p = rho*R_s*T, ideal gas law in specific form
# 2] h = u + p/rho , specific enthalpy of uniform system, where u=specific internal energy, p=pressure, and rho=density.
# 3] u = Cv*T, specific in ternal energy


class FluidBase:
    # collects Sym objects created by the fluid model so they can be included in diagram processing
    syms = []


class IdealGasAir(FluidBase):
    """
    ideal gas model for air. equations from:
    https://modelica.org/events/Conference2003/papers/h40_Elmqvist_fluid.pdf
    """

    def __init__(
        self,
        ev,
        name="IdealGasAir",
        P_ic=101325.0,
        T_ic=300.0,
    ):
        self.name = name

        # constants
        self.cp = 1006.0  # J/(kg*K)
        self.Rs_air = 287.052874
        self.Tref = 0.0
        self.href = 274648.7

        # HACK: this is super hacky for now. I made this because we need some values
        # to initialize fluid BaseProp variables such that they dont get the default
        # weak_ic of 0.0. for exmaple, for density, a weak_ic of 0.0 results in div-by-zero
        # in the IC solving process.
        h_ic, u_ic, d_ic = self.get_h_u_d_ics(P_ic, T_ic)
        self.init = {"p": P_ic, "T": T_ic, "h": h_ic, "u": u_ic, "d": d_ic}

    def gen_eqs(self, ev, p, h, T, u, d):
        eqs = [
            sp.Eq(p, d * self.Rs_air * T),
            sp.Eq(h, self.href + self.cp * T),
            sp.Eq(u, h - p / d),
        ]
        return eqs

    def get_T_u_d_ics(self, p, h):
        T = (h - self.href) / self.cp
        d = p / (self.Rs_air * T)
        u = h - p / d

        return T, u, d

    def get_h_u_d_ics(self, p, T):
        h = T * self.cp + self.href
        d = p / (self.Rs_air * T)
        u = h - p / d

        return h, u, d


class WaterLiquidSimple(FluidBase):
    """
    Class for Liquid Water with simple state equations

    The only option for state variabes is: p,h
    """

    def __init__(
        self,
        ev,
        name="WaterLiquid",
        P_ic=101325.0,
        T_ic=300.0,
    ):
        self.name = name

        # constants
        self.cp = 4180.0  # J/(kg*K)
        self.cv = 4130.0  # J/(kg*K)
        self.density = 997  # kg/m3
        # T2h is a linear approximation from the graph here: https://www.engineeringtoolbox.com/water-properties-d_1508.html
        # self.T2h = lambda T: (T - 273.15) * 1350e3 / 573  # (J/kg)/(K)
        # HACK: this function was intentionally crafted identical to the u=cp*T equation so that h and u are both identical
        # proxies for T, thus making the Fluid not fully thermodynamic, but a simplified thermo fluid, whihch sort of what
        # we were aiming for anyway.
        self.T2h = lambda T: T * self.cp  # (J/kg)/(K)

        # HACK: this is super hacky for now. I made this because we need some values
        # to initialize fluid BaseProp variables such that they dont get the default
        # weak_ic of 0.0. for example, for density, a weak_ic of 0.0 results in div-by-zero
        # in the IC solving process.
        h_ic, u_ic, d_ic = self.get_h_u_d_ics(P_ic, T_ic)
        self.init = {"p": P_ic, "T": T_ic, "h": h_ic, "u": 0.0, "d": d_ic}

    def gen_eqs(self, ev, p, h, T, u, d):
        # NOTE: these equations are wrong, but they allow simple modeling of incompressible fluid
        eqs = [
            sp.Eq(d, self.density),
            sp.Eq(u, T * self.cp),
            sp.Eq(h, self.T2h(T)),
        ]
        return eqs

    def get_h_u_d_ics(self, p, T):
        d = self.density
        u = T * self.cp
        h = self.T2h(T)

        return h, u, d


class WaterLiquid(FluidBase):
    """
    Class for Liquid Water based on region 1 of IF97 standard.
    http://www.iapws.org/relguide/IF97-Rev.pdf

    The only option for state variabes is: p,h
    """

    def __init__(
        self,
        ev,
        name="WaterLiquid",
        P_ic=101325.0,
        T_ic=300.0,
    ):
        self.name = name

        # constants
        self.cp = 4180.0  # J/(kg*K)
        self.cv = 4130.0  # J/(kg*K)

        # HACK: gamma_nonce is used to create unique symbol names for the
        # gibbs free energy variables used in the state equations. unique names
        # are required each time the state equations are added by a component to
        # complete the components port state fluid properties. Of course, the
        # 'right' way to achieve this is to have the component pass in a symbol
        # name prefix, e.g. '<cmp_name>_<port_name>_' like we do for p,T,h,u,d, but
        # this is deviation from the interface of IdealGas that doesn't need this.
        # maybe in the future we can find a better solution for naming these, but
        # for now this will work.
        self.gamma_nonce = 0

        # coefficients from table 2. auto linter forces them to be formatted annoyingly
        self.Iis = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                4,
                5,
                8,
                8,
                21,
                23,
                29,
                30,
                31,
                32,
            ],
            dtype=np.float64,
        )
        self.Jis = np.array(
            [
                -2,
                -1,
                0,
                1,
                2,
                3,
                4,
                5,
                -9,
                -7,
                -1,
                0,
                1,
                3,
                -3,
                0,
                1,
                3,
                17,
                -4,
                0,
                6,
                -5,
                -2,
                10,
                -8,
                -11,
                -6,
                -29,
                -31,
                -38,
                -39,
                -40,
                -41,
            ],
            dtype=np.float64,
        )
        self.nis = np.array(
            [
                0.14632971213167,
                -0.84548187169114,
                -0.37563603672040e1,
                0.33855169168385e1,
                -0.95791963387872,
                0.15772038513228,
                -0.16616417199501e-1,
                0.81214629983568e-3,
                0.28319080123804e-3,
                -0.60706301565874e-3,
                -0.18990068218419e-1,
                -0.32529748770505e-1,
                -0.21841717175414e-1,
                -0.52838357969930e-4,
                -0.47184321073267e-3,
                -0.30001780793026e-3,
                0.47661393906987e-4,
                -0.44141845330846e-5,
                -0.72694996297594e-15,
                -0.31679644845054e-4,
                -0.28270797985312e-5,
                -0.85205128120103e-9,
                -0.22425281908000e-5,
                -0.65171222895601e-6,
                -0.14341729937924e-12,
                -0.40516996860117e-6,
                -0.12734301741641e-8,
                -0.17424871230634e-9,
                -0.68762131295531e-18,
                0.14478307828521e-19,
                0.26335781662795e-22,
                -0.11947622640071e-22,
                0.18228094581404e-23,
                -0.93537087292458e-25,
            ],
            dtype=np.float64,
        )
        self.R = 0.461526e3  # 0.461526 kJ/(kg*K) in J/(kg*K)
        self.pstar = 16.53e6  # 16.53MPa in Pa
        self.Tstar = 1386

        # HACK: this is super hacky for now. I made this because we need some values
        # to initialize fluid BaseProp variables such that they dont get the default
        # weak_ic of 0.0. for exmaple, for density, a weak_ic of 0.0 results in div-by-zero
        # in the IC solving process.
        h_ic, u_ic, d_ic = self.get_h_u_d_ics(P_ic, T_ic)
        self.init = {"p": P_ic, "T": T_ic, "h": h_ic, "u": 0.0, "d": d_ic}

    def gen_eqs(self, ev, p, h, T, u, d, return_gamma=False):
        pi = p / self.pstar
        tau = self.Tstar / T

        gpnam = "waterif97_gp" + str(self.gamma_nonce)
        gtnam = "waterif97_gt" + str(self.gamma_nonce)

        gp = sp.Symbol(gpnam)
        gt = sp.Symbol(gtnam)

        base_name = "waterIF97R1_inst" + str(self.gamma_nonce)
        self.gamma_nonce += 1  # increment
        gpnam = "gp"
        gtnam = "gt"
        kind = SymKind.var
        gp = Sym(
            ev,
            sym_name=gpnam,
            base_name=base_name,
            kind=kind,
        )
        gt = Sym(
            ev,
            sym_name=gtnam,
            base_name=base_name,
            kind=kind,
        )
        self.syms.append(gp)
        self.syms.append(gt)

        gpterms = []
        gtterms = []
        for Ii, Ji, ni in zip(self.Iis, self.Jis, self.nis):
            # expr for ith term of summation for gamma
            gterm1 = ni * (7.1 - pi) ** Ii
            gterm2 = (tau - 1.222) ** Ji

            # expr for ith term of summation for d(gamma)/d(pi)
            gpterm1 = -ni * Ii * (7.1 - pi) ** (Ii - 1)
            gpterms.append(gpterm1 * gterm2)

            # expr for ith term of summation for d(gamma)/d(tau)
            gtterm2 = Ji * (tau - 1.222) ** (Ji - 1)
            gtterms.append(gterm1 * gtterm2)

        eqs = [
            # gamma equations from table 4
            sp.Eq(gp.s, sp.core.add.Add(*gpterms)),
            sp.Eq(gt.s, sp.core.add.Add(*gtterms)),
            # water property equations from table 3, rearranged to have prop in LHS
            # not necessary, but convenient for testing.
            sp.Eq(d, p / (self.R * T * pi * gp.s)),  # density, i.e. 1/specific_vol
            sp.Eq(u, (tau * gt.s - pi * gp.s) * (self.R * T)),  # u
            sp.Eq(h, tau * gt.s * self.R * T),  # h
        ]

        if return_gamma:
            return eqs, gp.s, gt.s
        return eqs

    def get_h_u_d_ics(self, p, T):
        pi = p / self.pstar
        tau = self.Tstar / T
        g = 0.0
        gp = 0.0
        gt = 0.0
        for Ii, Ji, ni in zip(self.Iis, self.Jis, self.nis):
            # expr for ith term of summation for gamma
            gterm1 = ni * (7.1 - pi) ** Ii
            gterm2 = (tau - 1.222) ** Ji
            g += gterm1 * gterm2

            # expr for ith term of summation for d(gamma)/d(pi)
            gpterm1 = -ni * Ii * (7.1 - pi) ** (Ii - 1)
            gp += gpterm1 * gterm2

            # expr for ith term of summation for d(gamma)/d(tau)
            gtterm2 = Ji * (tau - 1.222) ** (Ji - 1)
            gt += gterm1 * gtterm2

        d = p / (self.R * T * pi * gp)
        u = (tau * gt - pi * gp) * (self.R * T)
        h = tau * gt * self.R * T

        return h, u, d
