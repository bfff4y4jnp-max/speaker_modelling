"""Reused-interior solver: overlap projection and agreement with the full coupled solve."""
import os

import numpy as np

from spkmod import Cabinet, Driver, Piston, SolverSettings, Wiring, solve_network, solve_scene
from spkmod.reduced import overlap_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "..", "examples", "demo", "cad", "sub.step")


def test_overlap_two_triangulations_of_a_square():
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = sq[[[0, 1, 2], [0, 2, 3]]]                     # diagonal one way
    E = sq[[[0, 1, 3], [1, 2, 3]]]                     # diagonal the other way
    A = overlap_matrix(F, E)
    assert np.allclose(A.sum(0), 0.5) and np.allclose(A.sum(1), 0.5) and np.allclose(A, 0.25)
    assert np.allclose(overlap_matrix(F, E + [0, 0, 0.01]), 0)       # parallel but not coplanar


def test_reuse_matches_full_for_touching_and_rotated_boxes():
    pis = lambda: [Piston("woofer", (0.491, 0.225, 0.330), (1, 0, 0), 0.258, cutout=0.28)]
    cabs = [Cabinet("a", step=SUB, pistons=pis()),
            Cabinet("b", step=SUB, pistons=pis(), position=(0, 0.45, 0)),                  # touching a
            Cabinet("c", step=SUB, pistons=pis(), position=(2.2, 0.2, 0), rotation=(0, 0, 150))]
    f = np.array([40.0, 90.0])
    pts = np.array([[4, 0.4, 1.6], [3, -2, 1.2], [-3, 1, 1.6]])
    r = {m: solve_scene(cabs, f, points=pts, ground=True, settings=SolverSettings(h_max=0.1, cache_dir=None, interior=m))
         for m in ("full", "reuse")}
    drv = Driver.from_ts("w", 35, 0.38, 5.0, 0.12, 5.6, 0.0522, Le=1.2e-3)
    wr = Wiring().add("a.woofer", drv, "x").add("b.woofer", drv, "x").add("c.woofer", drv, "y")
    n = {m: solve_network(r[m], wr) for m in r}
    for ch in ("x", "y"):
        d = 20 * np.log10(np.abs(n["reuse"][ch]["p"]) / np.abs(n["full"][ch]["p"]))
        assert np.abs(d).max() < 0.1
        assert np.allclose(np.abs(n["reuse"][ch]["Z"]), np.abs(n["full"][ch]["Z"]), rtol=0.03)
