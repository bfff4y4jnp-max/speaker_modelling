"""Air properties, atmospheric absorption and porous-material models.

Conventions: SI units throughout. Material models are written in the
engineering e^{+jwt} convention (as they appear in the literature); the
solvers, which work in e^{-iwt}, take complex conjugates where needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Air:
    temperature: float = 20.0      # deg C
    humidity: float = 50.0         # % RH
    pressure: float = 101325.0     # Pa

    @property
    def T(self) -> float:
        return self.temperature + 273.15

    @property
    def gamma(self) -> float:
        return 1.4

    @property
    def rho(self) -> float:
        return self.pressure / (287.058 * self.T)

    @property
    def c(self) -> float:
        return float(np.sqrt(self.gamma * 287.058 * self.T))

    @property
    def mu(self) -> float:
        """Dynamic viscosity (Sutherland)."""
        return 1.458e-6 * self.T ** 1.5 / (self.T + 110.4)

    @property
    def cp(self) -> float:
        return 1005.0

    @property
    def kappa(self) -> float:
        """Thermal conductivity."""
        return 2.646e-3 * self.T ** 1.5 / (self.T + 245.4 * 10 ** (-12 / self.T))

    @property
    def prandtl(self) -> float:
        return self.mu * self.cp / self.kappa

    def delta_v(self, f):
        """Viscous boundary-layer thickness sqrt(2 mu / (w rho))."""
        return np.sqrt(2 * self.mu / (2 * np.pi * np.asarray(f) * self.rho))

    def delta_t(self, f):
        """Thermal boundary-layer thickness."""
        return self.delta_v(f) / np.sqrt(self.prandtl)

    def absorption(self, f):
        """Atmospheric absorption, Np/m (ISO 9613-1)."""
        f = np.asarray(f, float)
        T, T0 = self.T, 293.15
        pr = self.pressure / 101325.0
        psat = 10 ** (-6.8346 * (273.16 / T) ** 1.261 + 4.6151)
        h = self.humidity * psat / pr
        frO = pr * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
        frN = pr * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
        a_db = 8.686 * f ** 2 * (
            1.84e-11 / pr * (T / T0) ** 0.5
            + (T / T0) ** -2.5 * (0.01275 * np.exp(-2239.1 / T) / (frO + f ** 2 / frO)
                                  + 0.1068 * np.exp(-3352.0 / T) / (frN + f ** 2 / frN)))
        return a_db / 8.686


def miki(f, sigma, air: Air = Air()):
    """Miki (1990) equivalent fluid for fibrous fill / ground.

    Returns (rho_eff, c_eff) in the e^{+jwt} convention.
    sigma: static flow resistivity, Pa s/m^2.
    """
    f = np.asarray(f, float)
    X = f / sigma
    rho0, c0 = air.rho, air.c
    Zc = rho0 * c0 * (1 + 0.0699 * X ** -0.632 - 1j * 0.1071 * X ** -0.632)
    kc = (2 * np.pi * f / c0) * (1 + 0.1093 * X ** -0.618 - 1j * 0.1597 * X ** -0.618)
    w = 2 * np.pi * f
    rho_eff = Zc * kc / w
    c_eff = w / kc
    return rho_eff, c_eff


def ground_reflection(f, r2, cos_theta, sigma=200e3, air: Air = Air()):
    """Spherical-wave reflection coefficient Q for a locally reacting ground.

    r2: image path length, cos_theta: cosine of incidence angle from normal.
    Uses Miki surface impedance and the Chien-Soroka / Attenborough boundary
    loss factor F(w). Engineering convention. sigma=inf gives Q=1 (rigid).
    """
    f = np.asarray(f, float)
    if not np.isfinite(sigma):
        return np.ones(np.broadcast(f, r2).shape, complex)
    from scipy.special import wofz
    rho_e, c_e = miki(f, sigma, air)
    Zs = rho_e * c_e / (air.rho * air.c)          # normalised, semi-infinite
    k = 2 * np.pi * f / air.c
    beta = 1 / Zs
    Rp = (cos_theta - beta) / (cos_theta + beta)
    w = np.sqrt(1j * k * r2 / 2) * (cos_theta + beta)
    F = 1 + 1j * np.sqrt(np.pi) * w * wofz(w)
    return Rp + (1 - Rp) * F
