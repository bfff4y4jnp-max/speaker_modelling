"""End-to-end coupled FEM-BEM checks against lumped theory (slow: ~2 min each)."""
import numpy as np

from spkmod import Air, Cabinet, Driver, Piston, SolverSettings, Wiring, combine, hollow_box, solve_network, solve_scene, tube
from spkmod.system import log_freqs

W, H, D, T = 0.4, 0.4, 0.5, 0.018
S = SolverSettings(h_max=0.08, cache_dir=None)


def test_sealed_box_stiffness_and_monopole():
    cab = Cabinet("box", builder=hollow_box((W, H, D), T, holes=[((W / 2, H / 2, D - T / 2), (0, 0, 1), 0.24, 0.05)]),
                  pistons=[Piston("woofer", (W / 2, H / 2, D - T / 2), (0, 0, 1), 0.2, cutout=0.24)])
    res = solve_scene([cab], [20, 30, 50], directions=np.array([[0, 0, 1.0]]), ground=False, settings=S)
    air, Sd = Air(), np.pi * 0.1 ** 2
    w = 2 * np.pi * res.freqs
    X = (1j * w * res.Za[:, 0, 0]).real            # K - w^2 M
    K, M = np.linalg.solve(np.vstack([np.ones(2), -w[:2] ** 2]).T, X[:2])
    V = (W - 2 * T) * (H - 2 * T) * (D - 2 * T) + np.pi * 0.12 ** 2 * 0.005
    assert abs(K / (air.rho * air.c ** 2 * Sd ** 2 / V) - 1) < 0.01
    assert 0.003 < M < 0.01
    mono = w * air.rho * Sd / (4 * np.pi)
    assert np.all(np.abs(np.abs(res.T_far[:, 0, 0]) / mono - 1) < 0.015)


def test_vented_box_tuning():
    pd, pl = 0.07, 0.15
    cab = Cabinet("box", builder=combine(
        hollow_box((W, H, D), T, holes=[((W / 2, 0.28, D - T / 2), (0, 0, 1), 0.24, 0.05),
                                        ((W / 2, 0.09, D - T / 2), (0, 0, 1), pd, 0.05)]),
        tube((W / 2, 0.09, D - pl / 2), (0, 0, 1), pd, pl, 0.006)),
        pistons=[Piston("woofer", (W / 2, 0.28, D - T / 2), (0, 0, 1), 0.2, cutout=0.24)])
    f = log_freqs(25, 45, 48)
    res = solve_scene([cab], f, ground=False, settings=S)
    drv = Driver.from_ts("w", Fs=30, Qes=0.4, Qms=4, Vas=0.08, Re=6, Sd=np.pi * 0.01)
    net = solve_network(res, Wiring().add("box.woofer", drv, "ch"))["ch"]
    fb_model = f[np.argmin(np.abs(net["x"][:, 0]))]
    air = Air()
    V = (W - 2 * T) * (H - 2 * T) * (D - 2 * T) - np.pi * (pd / 2 + 0.006) ** 2 * (pl - T)
    a = pd / 2
    fb_lumped = [air.c / (2 * np.pi) * np.sqrt(np.pi * a ** 2 / (V * (pl + ec * a))) for ec in (1.7, 1.46)]
    assert fb_lumped[0] * 0.97 < fb_model < fb_lumped[1] * 1.03
