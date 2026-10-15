"""Interior acoustic FEM (scikit-fem, quadratic tetrahedra).

Internal convention e^{-iwt}. Weak form, region r with (rho_r, c_r):
    sum_r  1/rho_r (grad p, grad q) - w^2/(rho_r c_r^2) (p, q)
        + beta/rho [dv (gradT p, gradT q) + (gamma-1) dt k^2 (p, q)]_walls
        = i w  sum_faces  u_out (1, q)
where u_out is the normal velocity of the boundary *out of* the fluid and
the wall term is the Berggren et al. (2018) viscothermal boundary-layer
condition, beta = -(1+i)/2 in this time convention.

Facet data are piecewise constant. For P2 on a triangle the vertex shape
functions integrate to zero and each edge function to area/3, so facet
couplings are assembled exactly without quadrature.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from skfem import Basis, BilinearForm, ElementTetP2, FacetBasis, MeshTet, asm
from skfem.helpers import dot, grad

from .air import Air, miki


@BilinearForm
def _stiff(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def _mass(u, v, w):
    return u * v


@BilinearForm
def _tan_stiff(u, v, w):
    n = w.n
    gu, gv = grad(u), grad(v)
    gu = gu - dot(gu, n) * n
    gv = gv - dot(gv, n) * n
    return dot(gu, gv)


class InteriorFEM:
    """Helmholtz FEM on the air cavities of a scene.

    nodes (N,3), tets (M,4), tet_region (M,) region ids; regions maps id ->
    flow resistivity (None/0 for plain air).
    wall_tris (k,3): rigid wall triangles that get boundary-layer losses.
    """

    def __init__(self, nodes, tets, tet_region=None, regions=None, wall_tris=None, air: Air = Air(),
                 losses=True):
        self.air = air
        tets = np.asarray(tets, np.int64)
        used = np.unique(tets)
        self._loc = np.full(len(nodes), -1, np.int64)       # global node -> FEM node
        self._loc[used] = np.arange(len(used))
        self.mesh = MeshTet(np.asarray(nodes, float)[used].T.copy(), self._loc[tets].T.copy())
        self.elem = ElementTetP2()
        self.basis = Basis(self.mesh, self.elem, intorder=4)
        self.ndof = self.basis.N
        M = len(tets)
        tet_region = np.zeros(M, int) if tet_region is None else np.asarray(tet_region)
        self.regions = {}
        for r in np.unique(tet_region):
            idx = np.where(tet_region == r)[0]
            b = Basis(self.mesh, self.elem, elements=idx, intorder=4)
            sigma = (regions or {}).get(int(r))
            self.regions[int(r)] = (sigma, asm(_stiff, b), asm(_mass, b))
        # edge lookup for exact facet integrals
        e = self.mesh.edges
        self._edge_id = {(int(a), int(b)): i for i, (a, b) in enumerate(np.sort(e, 0).T)}
        self._fac_id = {tuple(f): i for i, f in enumerate(np.sort(self.mesh.facets, 0).T)}
        self.losses = losses
        self.wall_K = self.wall_M = None
        if losses and wall_tris is not None and len(wall_tris):
            fac = self.facets(wall_tris)
            fb = FacetBasis(self.mesh, self.elem, facets=fac, intorder=4)
            self.wall_K = asm(_tan_stiff, fb)
            self.wall_M = asm(_mass, fb)

    # ------------------------------------------------------------- facets
    def facets(self, tris):
        try:
            return np.array([self._fac_id[tuple(sorted(map(int, t)))] for t in self._loc[np.asarray(tris)]],
                            np.int64)
        except KeyError as exc:
            raise ValueError("triangle is not a facet of the FEM mesh") from exc

    def facet_matrix(self, tris):
        """C (ndof x ntri): column e holds the integrals of the P2 basis over triangle e."""
        tris = self._loc[np.asarray(tris, np.int64)]
        if (tris < 0).any():
            raise ValueError("triangle is not on the FEM mesh")
        p = self.mesh.p.T
        area = 0.5 * np.linalg.norm(np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]]), axis=1)
        ed = self.basis.edge_dofs[0] if self.basis.edge_dofs.ndim == 2 else self.basis.edge_dofs
        rows, cols, vals = [], [], []
        for j, (a, b, c) in enumerate(tris):
            for u, v in ((a, b), (b, c), (a, c)):
                rows.append(ed[self._edge_id[(min(u, v), max(u, v))]])
                cols.append(j)
                vals.append(area[j] / 3)
        return sp.csc_matrix((vals, (rows, cols)), shape=(self.ndof, len(tris)))

    # ------------------------------------------------------------- operator
    def operator(self, f):
        """System matrix S(w) (complex, CSC)."""
        air = self.air
        w = 2 * np.pi * f
        S = None
        for sigma, K, M in self.regions.values():
            if sigma:
                rho_e, c_e = miki(f, sigma, air)
                rho_e, c_e = np.conj(rho_e), np.conj(c_e)        # to e^{-iwt}
            else:
                rho_e, c_e = air.rho, air.c
            term = K / rho_e - (w ** 2 / (rho_e * c_e ** 2)) * M
            S = term if S is None else S + term
        if self.wall_K is not None:
            beta = -(1 + 1j) / 2
            k = w / air.c
            S = S + (beta / air.rho) * (air.delta_v(f) * self.wall_K
                                        + (air.gamma - 1) * air.delta_t(f) * k ** 2 * self.wall_M)
        return sp.csc_matrix(S, dtype=complex)

    def point_matrix(self, points):
        """Interpolation matrix (npts x ndof) for points inside the cavity."""
        return self.basis.probes(np.asarray(points, float).T)
