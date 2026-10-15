"""Interior FEM: closed tube driven by a piston, with and without wall losses."""
import numpy as np
import scipy.sparse.linalg as spla
import pytest

from spkmod.air import Air
from spkmod.fem import InteriorFEM


def tube_mesh(L=1.0, a=0.04, h=0.012):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.addBox(0, 0, 0, L, a, a)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.mesh.generate(3)
    tags, x, _ = gmsh.model.mesh.getNodes()
    idx = {int(t): i for i, t in enumerate(tags)}
    nodes = x.reshape(-1, 3)
    _, tt = gmsh.model.mesh.getElementsByType(4)
    tets = np.vectorize(idx.get)(tt.reshape(-1, 4))
    tris = []
    for _, s in gmsh.model.getEntities(2):
        _, tn = gmsh.model.mesh.getElementsByType(2, s)
        tris.append(np.vectorize(idx.get)(tn.reshape(-1, 3)))
    gmsh.finalize()
    tris = np.vstack(tris)
    cx = nodes[tris][:, :, 0]
    piston = tris[(cx < 1e-9).all(1)]
    walls = tris[~(cx < 1e-9).all(1) & ~(cx > L - 1e-9).all(1)]
    return nodes, tets, piston, walls


def analytic(f, L, a, air, losses):
    w = 2 * np.pi * f
    k = w / air.c
    PA = 4 * a / a ** 2
    if losses:
        rv = 1 + (1 + 1j) / 2 * PA * air.delta_v(f)
        rk = 1 + (1 + 1j) / 2 * PA * (air.gamma - 1) * air.delta_t(f)
    else:
        rv = rk = 1.0
    kc = k * np.sqrt(rv * rk)
    A = 1j * w * air.rho * rv / (kc * np.sin(kc * L))
    return A * np.cos(kc * L)


@pytest.mark.parametrize("losses", [False, True])
def test_tube(losses):
    L, a = 1.0, 0.04
    air = Air()
    nodes, tets, piston, walls = tube_mesh(L, a)
    fem = InteriorFEM(nodes, tets, wall_tris=walls, air=air, losses=losses)
    C = fem.facet_matrix(piston)
    Pp = fem.point_matrix(np.array([[0.0, a / 2, a / 2]]))
    f0 = air.c / (2 * L)
    for f in [0.6 * f0, f0 * 0.999, 1.37 * f0]:
        w = 2 * np.pi * f
        rhs = 1j * w * (C @ -np.ones(C.shape[1]))     # piston moves into the fluid: u_out = -1
        p = spla.spsolve(fem.operator(f), rhs)
        pf = (Pp @ p)[0]
        pa = analytic(f, L, a, air, losses)
        assert abs(pf / pa - 1) < 0.03, (f, pf, pa)
