"""Baffled-mouth exterior: a tube ending flush in an infinite baffle, driven by a piston at its bottom."""
import numpy as np
from scipy.special import j1, struve

from spkmod import Air, Cabinet, Piston, SolverSettings, solve_scene


def _block_with_tube(a, L, t, H=0.2):
    def build(occ):
        b = occ.addBox(-0.15, -0.15, 0, 0.3, 0.3, H)
        c = occ.addCylinder(0, 0, H - L - t, 0, 0, L + t + 0.01, a)
        out, _ = occ.cut([(3, b)], [(3, c)])
        return [x for d, x in out if d == 3]
    return build


def test_piston_in_baffle_impedance_and_directivity():
    """Piston 2 mm below the mouth of a flush hole: the exact baffled-piston solution applies."""
    a, L, t, H = 0.05, 0.002, 0.008, 0.2
    cab = Cabinet("wg", builder=_block_with_tube(a, L, t, H),
                  pistons=[Piston("hf", (0, 0, H - L - t / 2), (0, 0, 1), 2 * a, cutout=2 * a, thickness=t)])
    air = Air()
    k = np.array([0.5, 1.0, 2.0, 3.0]) / a
    f = k * air.c / (2 * np.pi)
    th = np.radians(30)
    dirs = np.array([[0, 0, 1.0], [np.sin(th), 0, np.cos(th)], [0, 0, -1.0]])
    res = solve_scene([cab], f, directions=dirs, ground=False, air=air,
                      settings=SolverSettings(h_max=0.02, cache_dir=None, exterior="baffle", losses=False,
                                              elements_per_wavelength=10))
    S, rc = np.pi * a ** 2, air.rho * air.c
    x = 2 * k * a
    ZL = rc * S * ((1 - 2 * j1(x) / x) + 1j * 2 * struve(1, x) / x)       # baffled piston, e^{+jwt}
    tn = np.tan(k * L)
    Zin = rc * S * (ZL + 1j * rc * S * tn) / (rc * S + 1j * ZL * tn)     # + the 2 mm air layer
    assert np.all(np.abs(res.Za[:, 0, 0] / Zin - 1) < 0.03)
    w = 2 * np.pi * f
    U = np.abs(res.U_ap[:, 0, 0])
    assert np.allclose(np.abs(res.T_far[:, 0, 0]), w * air.rho * U / (2 * np.pi), rtol=1e-6)   # |p| r on axis
    xs = k * a * np.sin(th)
    D = np.abs(res.T_far[:, 1, 0]) / np.abs(res.T_far[:, 0, 0])
    assert np.all(np.abs(D - np.abs(2 * j1(xs) / xs)) < 0.02)                                  # piston directivity
    assert np.all(res.T_far[:, 2, 0] == 0)                                                     # behind the baffle
