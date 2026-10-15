"""Piston recess (cone dish): the dish air joins the chamber in front of the cone."""
import numpy as np
import pytest

from spkmod import Air, Cabinet, Piston, SolverSettings, solve_scene
from spkmod.geometry import build_scene_mesh

W, T = 0.3, 0.018
ZD, TD = 0.100, 0.018          # internal divider: z 0.100 .. 0.118
R_P, CUT, TH, WALL = 0.075, 0.17, 0.008, 0.005


def _two_chamber(occ):
    o = occ.addBox(0, 0, 0, W, W, 0.5)
    i = occ.addBox(T, T, T, W - 2 * T, W - 2 * T, 0.5 - 2 * T)
    bx, _ = occ.cut([(3, o)], [(3, i)])
    div = occ.addBox(T, T, ZD, W - 2 * T, W - 2 * T, TD)
    hole = occ.addCylinder(W / 2, W / 2, ZD - 0.01, 0, 0, TD + 0.02, CUT / 2)
    dv, _ = occ.cut([(3, div)], [(3, hole)])
    return [t for d, t in bx + dv if d == 3]


def _stiffness(recess):
    cab = Cabinet("b", builder=_two_chamber,
                  pistons=[Piston("d", (W / 2, W / 2, ZD + TD / 2), (0, 0, -1), 2 * R_P, cutout=CUT, thickness=TH,
                                  recess=recess, basket_wall=WALL)])
    f = np.array([15.0, 25.0])
    res = solve_scene([cab], f, ground=False, settings=SolverSettings(h_max=0.06, cache_dir=None, losses=False))
    w = 2 * np.pi * f
    X = (1j * w * res.Za[:, 0, 0]).real                     # K - w^2 M
    return np.linalg.solve(np.vstack([np.ones(2), -w ** 2]).T, X)[0]


@pytest.mark.parametrize("recess", [0.0, 0.03])
def test_recess_moves_dish_air_to_the_front_chamber(recess):
    air = Air()
    S = np.pi * R_P ** 2
    inner = W - 2 * T
    # front chamber: below the divider + the hole in front of the piston (dish when recessed)
    zc = ZD + TD / 2
    A_hole = np.pi * (CUT / 2) ** 2                          # baffle hole outside the frame's thickness is air
    front = inner * inner * (ZD - T) + A_hole * ((zc - TH / 2) - ZD) + S * recess
    rear = inner * inner * (0.5 - T - (ZD + TD)) + A_hole * ((ZD + TD) - (zc + TH / 2))
    if recess:          # piston, dish and basket fill the hole behind the frame and stick out of the divider back
        rear -= A_hole * ((ZD + TD) - (zc + TH / 2))
        protr = (zc + recess + TH / 2) - (ZD + TD)
        rear -= np.pi * (R_P + WALL) ** 2 * max(protr, 0)
    K_th = air.rho * air.c ** 2 * S ** 2 * (1 / front + 1 / rear)
    K = _stiffness(recess)
    assert abs(K / K_th - 1) < 0.02, (K, K_th)


def test_recess_meshes_and_cone_depth_alias():
    p = Piston("d", (0, 0, 0), (0, 0, 1), 0.2, cone_depth=0.09)
    assert abs(p.recess_depth - 0.03) < 1e-12
    cab = Cabinet("b", builder=_two_chamber,
                  pistons=[Piston("d", (W / 2, W / 2, ZD + TD / 2), (0, 0, -1), 2 * R_P, cutout=CUT, recess=0.045)])
    assert len(build_scene_mesh([cab], 0.08, ground=False).tets)
