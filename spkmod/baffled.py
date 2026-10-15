"""High-frequency model of a horn / waveguide whose mouth lies in a large flat face: the interior is the
reduced cabinet FEM (as for solver.interior: reuse) and the exterior is an infinite rigid baffle in the
mouth plane (Rayleigh / half-space Green's function). No exterior surface mesh is needed, so the cost is set
by the waveguide's own air volume and mouth area, not by the box.

Valid when the face around the mouth is several wavelengths across (the pole box's 330 x 260 mm face: from
roughly 2-3 kHz up); below that use the full-box BEM scene and split the frequency range between them.

Per frequency (e^{-iwt}), with the mouth facets as the coupling elements:
    (i w Q + i w rho R) v = -i w Qfb u,     R_ee' = 2 (G, 1)_{e x e'}   (Galerkin single layer, doubled)
    Za = -i w (Qfb^T v + Qbb),  p(x) = -2 sum_e (G(x, .), 1)_e  i w rho v_e   in front of the baffle, 0 behind.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .bem import _W6, _Q6, _bempp, _with_fallback
from .geometry import MeshError, _rotmat
from .reduced import _area_normal, cabinet_operator


def _plane(V):
    """Common plane of the mouth facets (local frame): point, unit normal (outward), max deviation."""
    a, n = _area_normal(V)
    nrm = (n * a[:, None]).sum(0)
    nrm /= np.linalg.norm(nrm)
    x0 = (V.mean(1) * a[:, None]).sum(0) / a.sum()
    dev = np.abs((V.reshape(-1, 3) - x0) @ nrm).max()
    return x0, nrm, dev


def _layer(points, V, k, far=False):
    """(npts x nfacets) integrals of G(x, y) = e^{ikr}/(4 pi r) over each facet (6-point rule);
    far=True: points are unit directions and the far-field kernel e^{-ik x.y}/(4 pi) is used."""
    a, _ = _area_normal(V)
    l1, l2 = _Q6[:, 0], _Q6[:, 1]
    q = V[:, 0, None] * (1 - l1 - l2)[None, :, None] + V[:, 1, None] * l1[None, :, None] + V[:, 2, None] * l2[None, :, None]
    w = a[:, None] * _W6[None]
    out = np.zeros((len(points), len(V)), complex)
    chunk = max(1, int(2e6 // (len(V) * 6)))
    for s in range(0, len(points), chunk):
        x = points[s:s + chunk]
        if far:
            G = np.exp(-1j * k * np.einsum("pd,mqd->pmq", x, q)) / (4 * np.pi)
        else:
            r = np.linalg.norm(x[:, None, None, :] - q[None], axis=-1)
            G = np.exp(1j * k * r) / (4 * np.pi * r)
        out[s:s + chunk] = (G * w[None]).sum(-1)
    return out


class _Rayleigh:
    """Galerkin matrix of the doubled single layer on the (open) mouth surface, via bempp-cl."""

    def __init__(self, V):
        b = _bempp()
        pts, inv = np.unique(np.round(V.reshape(-1, 3), 10), axis=0, return_inverse=True)
        tris = inv.reshape(-1, 3)
        self.grid = b.Grid(pts.T.copy(), tris.T.astype(np.uint32).copy())
        self.D0 = b.function_space(self.grid, "DP", 0)
        l2g = self.D0.local2global
        self.perm = np.array([l2g[e][0] for e in range(len(tris))])

    def matrix(self, k):
        from bempp_cl.api.operators.boundary import helmholtz as H
        D0 = self.D0
        S = _with_fallback(lambda d: H.single_layer(D0, D0, D0, k, device_interface=d))
        return 2 * S[np.ix_(self.perm, self.perm)]


def solve_band_baffled(cabinets, sub_bands, h, points, directions, image_points, air, settings):
    """Same output as scene._solve_band for a single cabinet radiating into a baffle at its mouth."""
    if len(cabinets) != 1:
        raise ValueError("exterior: baffle takes exactly one cabinet (the waveguide) per scene")
    cab = cabinets[0]
    rho, c = air.rho, air.c
    freqs = np.concatenate(sub_bands)
    op = cabinet_operator(cab, h, freqs, air, settings)
    Vl = op["ap_vertices"]
    if not len(Vl):
        raise MeshError(f"cabinet {cab.name}: no opening found for the baffled exterior")
    x0, n_l, dev = _plane(Vl)
    if dev > max(1e-4, 0.01 * h):
        raise MeshError(f"cabinet {cab.name}: its openings are not in one plane (max deviation {dev * 1000:.1f} mm). "
                        "For exterior: baffle the waveguide cabinet must open only through its flat face; give it an "
                        "envelope box aligned with that face (or a STEP in a frame where the face is axis-aligned).")
    R, T = _rotmat(cab.rotation), np.asarray(cab.position, float)
    Vg = Vl @ R.T + T                        # facets in the scene frame
    x0g, ng = R @ x0 + T, R @ n_l
    nb = len(op["names"])
    names = [f"{cab.name}.{p.name}" for p in cab.pistons]
    area = _area_normal(Vg)[0]
    ray = _Rayleigh(Vg)
    nf = len(freqs)
    out = {"names": names, "Za": np.zeros((nf, nb, nb), complex), "center": x0g, "ap_area": np.array([area.sum()]),
           "mesh": f"baffled mouth: {len(Vg)} facets in the plane through ({x0g[0]:.3f}, {x0g[1]:.3f}, {x0g[2]:.3f}) "
                   f"normal ({ng[0]:.3f}, {ng[1]:.3f}, {ng[2]:.3f}), h={h:.4f} m",
           "U_ap": np.zeros((nf, 1, nb), complex)}
    if points is not None:
        front = ((points - x0g) @ ng) > 0
        out["T"] = np.zeros((nf, len(points), nb), complex)
        if image_points:
            img = points * [1, 1, -1]
            front_img = ((img - x0g) @ ng) > 0
            out["T_img"] = np.zeros_like(out["T"])
    if directions is not None:
        dfront = (directions @ ng) > 0
        out["T_far"] = np.zeros((nf, len(directions), nb), complex)
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f
        k = w / c
        K = 1j * w * op["Q"][i] + 1j * w * rho * ray.matrix(k)
        Vv = sla.solve(K, -1j * w * op["Qfb"][i])               # mouth velocities per unit body velocity
        out["Za"][i] = -1j * w * (op["Qfb"][i].T @ Vv + op["Qbb"][i])
        out["U_ap"][i, 0] = area @ Vv
        g = 1j * w * rho * Vv
        if points is not None:
            out["T"][i] = np.where(front[:, None], -2 * _layer(points, Vg, k) @ g, 0)
            if image_points:
                out["T_img"][i] = np.where(front_img[:, None], -2 * _layer(img, Vg, k) @ g, 0)
        if directions is not None:
            out["T_far"][i] = np.where(dfront[:, None], -2 * _layer(directions, Vg, k, far=True) @ g, 0)
    for kk in ("Za", "T", "T_img", "T_far", "U_ap"):
        if out.get(kk) is not None:
            out[kk] = np.conj(out[kk])
    return out
