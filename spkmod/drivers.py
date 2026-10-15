"""Electro-mechanical models (engineering convention e^{+jwt}).

Every vibrating body exposes generalised velocity DOFs. A rigid piston
(cone driver, passive radiator) has one DOF; future bodies such as a
distributed-mode panel (plate modes + inertial exciter DOFs) plug in by
providing `Z(f)` (structural impedance matrix) and coil coupling vectors,
with the acoustic solve supplying the radiation impedance matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .air import Air


@dataclass
class Driver:
    """Thiele-Small driver. SI units (kg, m/N, N s/m, T m, H, ohm, m^2)."""
    name: str
    Re: float
    Bl: float
    Mms: float
    Cms: float
    Rms: float
    Sd: float
    Le: float = 0.0
    R2: float = 0.0          # LR-2 semi-inductance model (Klippel): R2 || L2 in series with Le
    L2: float = 0.0
    Mmd: Optional[float] = None
    Xmax: Optional[float] = None
    Pe: Optional[float] = None

    @classmethod
    def from_ts(cls, name, Fs, Qes, Qms, Vas, Re, Sd, Le=0.0, air: Air = Air(), **kw):
        """Build from Fs/Qes/Qms/Vas (Vas in m^3)."""
        Cms = Vas / (air.rho * air.c ** 2 * Sd ** 2)
        ws = 2 * np.pi * Fs
        Mms = 1 / (ws ** 2 * Cms)
        Rms = ws * Mms / Qms
        Bl = np.sqrt(ws * Mms * Re / Qes)
        return cls(name=name, Re=Re, Bl=Bl, Mms=Mms, Cms=Cms, Rms=Rms, Sd=Sd, Le=Le, **kw)

    def mmd(self, air: Air = Air()):
        """Moving mass without air load (the 3D solve supplies the air load).

        Small's convention: Mms = Mmd + 2 * 8 rho a^3 / 3.
        """
        if self.Mmd is not None:
            return self.Mmd
        a = np.sqrt(self.Sd / np.pi)
        return self.Mms - 2 * 8 * air.rho * a ** 3 / 3

    def Ze(self, f):
        jw = 2j * np.pi * np.asarray(f, float)
        Z = self.Re + jw * self.Le
        if self.R2 and self.L2:
            Z = Z + jw * self.L2 * self.R2 / (self.R2 + jw * self.L2)
        return Z

    def Zm(self, f, air: Air = Air()):
        jw = 2j * np.pi * np.asarray(f, float)
        return self.Rms + jw * self.mmd(air) + 1 / (jw * self.Cms)


@dataclass
class PassiveRadiator:
    name: str
    Mmd: float
    Cms: float
    Rms: float = 0.0

    def Zm(self, f, air: Air = Air()):
        jw = 2j * np.pi * np.asarray(f, float)
        return self.Rms + jw * self.Mmd + 1 / (jw * self.Cms)


@dataclass
class Wiring:
    """body name -> (driver/passive model, amp channel or None, polarity)."""
    entries: dict = field(default_factory=dict)

    def add(self, body, model, channel=None, polarity=1.0):
        self.entries[body] = (model, channel, polarity)
        return self


def solve_network(res, wiring: Wiring, air: Air = Air()):
    """Driven response of one solved scene for 1 V rms on each channel.

    res: SceneResult (engineering convention). Returns dict channel ->
    {'v': (nf, nbody) velocities, 'x': excursions, 'i': coil currents,
     'Z': channel input impedance, 'p': pressures at points, 'far': far field}.
    """
    f = res.freqs
    nb = len(res.body_names)
    models = []
    for name in res.body_names:
        if name not in wiring.entries:
            raise KeyError(f"body {name!r} has no driver/passive model in the wiring")
        models.append(wiring.entries[name])
    Zs = np.zeros((len(f), nb, nb), complex)
    Bl = np.zeros(nb)
    Ze = np.ones((len(f), nb), complex)
    for j, (m, ch, pol) in enumerate(models):
        Zs[:, j, j] = m.Zm(f, air)
        if isinstance(m, Driver) and ch is not None:
            Bl[j] = m.Bl * pol
            Ze[:, j] = m.Ze(f)
    Ztot = Zs + res.Za + np.einsum("j,fj,k->fjk", Bl, 1 / Ze, Bl) * np.eye(nb)[None]
    channels = sorted({ch for _, ch, _ in models if ch is not None})
    out = {}
    for ch in channels:
        e = np.array([1.0 if (c == ch and isinstance(m, Driver)) else 0.0 for m, c, _ in models])
        rhs = (Bl * e)[None, :] / Ze
        v = np.linalg.solve(Ztot, rhs[..., None])[..., 0]
        i = (e[None] - Bl[None] * v) / Ze * (e[None] != 0)
        jw = 2j * np.pi * f[:, None]
        d = {"v": v, "x": v / jw, "i": i, "Z": 1 / i.sum(1)}
        if res.T is not None:
            d["p"] = np.einsum("fpj,fj->fp", res.T, v)
        if res.T_img is not None:
            d["p_img"] = np.einsum("fpj,fj->fp", res.T_img, v)
        if res.T_far is not None:
            d["far"] = np.einsum("fpj,fj->fp", res.T_far, v)
        if res.U_ap is not None:
            d["U_ap"] = np.einsum("fgj,fj->fg", res.U_ap, v)
        out[ch] = d
    return out
