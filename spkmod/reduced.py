"""Reused-interior solver: each cabinet's interior FEM is solved once, in its own frame,
and reduced to an operator on its openings; layouts then only need the exterior BEM.

Per cabinet type and frequency (local frame, cached on disk):
    Q   = C_f^T S^-1 C_f     opening facets x opening facets
    Qfb = C_f^T S^-1 R_b     opening facets x body DOFs
    Qbb = R_b^T S^-1 R_b     body DOFs x body DOFs (interior load on the diaphragms)
with C_f the facet integrals of the P2 basis on the opening facets and R_b = -C_b G the
diaphragm loads. Identical cabinets (e.g. four deep30 horns) share one operator.

The scene is meshed on its surface only. Its opening triangles e are joined to the cabinet's
local opening facets f by the overlap areas A_fe (exact polygon clipping after mapping the
scene triangles into the cabinet frame): C' = C_f diag(1/area_f) A. Openings covered by a
neighbouring box or lying on the ground get no overlap and act as rigid walls, as in the full
solve. The coupled system per frequency (e^{-iwt}) is then dense and small:
    (i w A'Q A - i w rho E P_a) v_a = (i w rho E P_b G_bem - i w A'Qfb) u
    Za = -i w (N^T v_a + Qbb) + G_bem^T E_b p_s
"""
from __future__ import annotations

import dataclasses
import logging
import os
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from .bem import ExteriorBEM
from .fem import InteriorFEM
from .geometry import KIND_APERTURE, KIND_BODY, _rotmat, build_scene_mesh

log = logging.getLogger("spkmod")


# ----------------------------------------------------------------------------- geometry helpers
def _area_normal(P):
    cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    a = 0.5 * np.linalg.norm(cr, axis=1)
    return a, cr / np.maximum(2 * a[:, None], 1e-300)


def _clip(poly, a, b, inside_sign):
    """Sutherland-Hodgman: keep the part of poly on the inner side of edge a->b (2D)."""
    out = []
    n = len(poly)
    side = lambda p: inside_sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        sp_, sq = side(p), side(q)
        if sp_ >= 0:
            out.append(p)
        if (sp_ >= 0) != (sq >= 0):
            t = sp_ / (sp_ - sq)
            out.append(p + t * (q - p))
    return out


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    P = np.array(poly)
    return 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))


def overlap_matrix(F, E, tol=1e-5):
    """A[f, e] = area of (triangle F[f]) intersected with (triangle E[e]) for coplanar pairs. F, E: (n,3,3)."""
    aF, nF = _area_normal(F)
    aE, nE = _area_normal(E)
    lo_F, hi_F = F.min(1) - tol, F.max(1) + tol
    lo_E, hi_E = E.min(1), E.max(1)
    A = np.zeros((len(F), len(E)))
    for e in range(len(E)):
        cand = np.where((hi_F >= lo_E[e]).all(1) & (lo_F <= hi_E[e]).all(1)
                        & (np.abs(nF @ nE[e]) > 0.999)
                        & (np.abs(np.einsum("ij,ij->i", nF, E[e][0] - F[:, 0])) < tol))[0]
        if not len(cand):
            continue
        ax = np.argsort(np.abs(nE[e]))[:2]             # project on the two least-normal axes
        pe = [E[e][k][ax] for k in range(3)]
        for f in cand:
            pf = F[f][:, ax]
            sgn = np.sign((pf[1, 0] - pf[0, 0]) * (pf[2, 1] - pf[0, 1]) - (pf[1, 1] - pf[0, 1]) * (pf[2, 0] - pf[0, 0]))
            poly = pe
            for k in range(3):
                poly = _clip(poly, pf[k], pf[(k + 1) % 3], sgn)
                if not poly:
                    break
            A[f, e] = _poly_area(poly)
    return A


# ----------------------------------------------------------------------------- body shapes
def _shapes(nodes, tris, ids, centers, axes, nb, outward=False):
    G = np.zeros((len(tris), nb))
    if not len(tris):
        return G, np.zeros(0)
    a, nrm = _area_normal(nodes[tris])
    if not outward:
        cen = nodes[tris].mean(1)
        nrm = nrm * np.sign(np.einsum("ij,ij->i", nrm, cen - centers[ids]))[:, None]
    G[np.arange(len(tris)), ids] = np.einsum("ij,ij->i", nrm, axes[ids])
    return G, a


def _side_scale(G, a, true_area):
    """Rescale each body's faces so every meshed side carries exactly its true area."""
    scale = np.ones(G.shape[1])
    for j in range(G.shape[1]):
        col = G[:, j]
        sides = int((col > 0.5).any()) + int((col < -0.5).any())
        meshed = (np.abs(col) * a).sum()
        if sides and meshed > 0:
            scale[j] = sides * true_area[j] / meshed
    return G * scale


# ----------------------------------------------------------------------------- cabinet operator
def _cab_key(cab, h, air, settings, freqs):
    from .scene import _file_hash, _hash
    return _hash((_file_hash(cab.step) if cab.step else repr(cab.builder), repr(cab.pistons), repr(cab.stuffing),
                  cab.envelope, round(float(h), 7), repr(air), settings.losses, settings.h_min,
                  settings.piston_divisions, np.round(freqs, 5).tolist(), 1))


def cabinet_operator(cab, h, freqs, air, settings):
    """Reduced interior operator of one cabinet in its local frame (cached)."""
    from .scene import _atomic_savez
    key = _cab_key(cab, h, air, settings, freqs)
    path = os.path.join(settings.cache_dir, f"cabop_{key}.npz") if settings.cache_dir and cab.step else None
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    local = dataclasses.replace(cab, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))
    mesh = build_scene_mesh([local], h, settings.h_min, ground=False, piston_divisions=settings.piston_divisions)
    nodes, nb = mesh.nodes, len(mesh.body_names)
    ap_tris = mesh.bem_tris[mesh.bem_kind == KIND_APERTURE]
    G, a = _shapes(nodes, mesh.body_tris, mesh.body_ids, mesh.body_centers, mesh.body_axes, nb)
    G = _side_scale(G, a, mesh.body_area)
    nf, nfreq = len(ap_tris), len(freqs)
    Q = np.zeros((nfreq, nf, nf), complex)
    Qfb = np.zeros((nfreq, nf, nb), complex)
    Qbb = np.zeros((nfreq, nb, nb), complex)
    if len(mesh.tets):
        fem = InteriorFEM(nodes, mesh.tets, mesh.tet_region, mesh.regions, mesh.wall_tris, air, settings.losses)
        C_f = fem.facet_matrix(ap_tris).toarray() if nf else np.zeros((fem.ndof, 0))
        R_b = -(fem.facet_matrix(mesh.body_tris) @ G) if len(mesh.body_tris) else np.zeros((fem.ndof, nb))
        rhs = np.hstack([C_f, R_b]).astype(complex)
        for i, f in enumerate(freqs):
            X = spla.splu(fem.operator(f)).solve(rhs)
            Q[i] = C_f.T @ X[:, :nf]
            Qfb[i] = C_f.T @ X[:, nf:]
            Qbb[i] = R_b.T @ X[:, nf:]
    out = {"names": np.array(mesh.body_names), "ap_vertices": nodes[ap_tris], "ap_area": _area_normal(nodes[ap_tris])[0],
           "Q": Q, "Qfb": Qfb, "Qbb": Qbb, "freqs": np.asarray(freqs, float)}
    log.info("cabinet %s interior operator: %d opening facets, %d tets, %d freqs, %.0f s",
             cab.name, nf, len(mesh.tets), nfreq, time.time() - t0)
    if path:
        os.makedirs(settings.cache_dir, exist_ok=True)
        _atomic_savez(path, **out)
    return out


# ----------------------------------------------------------------------------- scene solve
def solve_band_reduced(cabinets, sub_bands, h, points, directions, ground, image_points, air, settings):
    """Same output as scene._solve_band, using reduced cabinet operators."""
    from .scene import _tri_area_normal
    import scipy.sparse as sp
    rho, c = air.rho, air.c
    freqs = np.concatenate(sub_bands)
    mesh = build_scene_mesh(cabinets, h, settings.h_min, ground=ground, piston_divisions=settings.piston_divisions,
                            surface_only=True)
    nodes, nb = mesh.nodes, len(mesh.body_names)
    ap = mesh.bem_kind == KIND_APERTURE
    bx = mesh.bem_kind == KIND_BODY
    ap_tris, bx_tris = mesh.bem_tris[ap], mesh.bem_tris[bx]
    n_a = len(ap_tris)
    active = np.concatenate([np.where(ap)[0], np.where(bx)[0]])
    G_bem, a_bx = _shapes(nodes, bx_tris, mesh.bem_body[bx], mesh.body_centers, mesh.body_axes, nb, outward=True)
    G_bem = _side_scale(G_bem, a_bx, mesh.body_area)

    # cabinet operators (shared between identical cabinets) and opening overlaps
    ops, cache = [], {}
    for cab in cabinets:
        k = _cab_key(cab, h, air, settings, freqs)
        if k not in cache:
            cache[k] = cabinet_operator(cab, h, freqs, air, settings)
        ops.append(cache[k])
    body_index = {n: i for i, n in enumerate(mesh.body_names)}
    blocks = []                                   # (scene opening rows, A^T D, body columns, op)
    for ci, (cab, op) in enumerate(zip(cabinets, ops)):
        rows = np.where(mesh.ap_cab == ci)[0]
        # shared operators keep the first cabinet's names: map this cabinet's bodies by piston order
        cols = np.array([body_index[f"{cab.name}.{p.name}"] for p in cab.pistons], int)
        R, T = _rotmat(cab.rotation), np.asarray(cab.position, float)
        E_loc = (nodes[ap_tris[rows]] - T) @ R            # global -> local: R^T (x - T)
        A = overlap_matrix(op["ap_vertices"], E_loc) if len(rows) and len(op["ap_vertices"]) else np.zeros((len(op["ap_vertices"]), len(rows)))
        if len(rows):
            cover = A.sum(0) / _area_normal(E_loc)[0]
            if cover.min() < 0.98:
                log.warning("cabinet %s: %d scene opening triangles only %.0f%% covered by its local openings",
                            cab.name, int((cover < 0.98).sum()), 100 * cover.min())
        AtD = A.T / np.maximum(op["ap_area"], 1e-300)[None, :]
        blocks.append((rows, AtD, A, cols, op))

    bem = ExteriorBEM(nodes, mesh.bem_tris, active, ground=0.0 if ground else None, quad_order=settings.quad_order)
    red = np.full(len(nodes), -1)
    red[bem.used] = np.arange(bem.n)

    def E_matrix(tris):
        a, _, _ = _tri_area_normal(nodes, tris)
        r = np.repeat(np.arange(len(tris)), 3)
        return sp.csr_matrix((np.repeat(a / 3, 3), (r, red[tris].ravel())), shape=(len(tris), bem.n))

    E_a, E_b = E_matrix(ap_tris), E_matrix(bx_tris)
    groups = np.unique(mesh.ap_group) if n_a else np.zeros(0, int)
    a_ap = _tri_area_normal(nodes, ap_tris)[0] if n_a else np.zeros(0)
    Ug = sp.csr_matrix((a_ap, (np.searchsorted(groups, mesh.ap_group), np.arange(n_a))),
                       shape=(len(groups), n_a)) if n_a else None
    nf = len(freqs)
    out = {"names": mesh.body_names, "Za": np.zeros((nf, nb, nb), complex),
           "center": nodes[np.unique(mesh.bem_tris)].mean(0), "ap_area": np.asarray(Ug.sum(1)).ravel() if Ug is not None else None,
           "mesh": mesh.summary() + " [reused interiors: " + ", ".join(
               f"{cab.name}:{len(op['ap_vertices'])} facets" for cab, op in zip(cabinets, ops)) + "]"}
    if points is not None:
        out["T"] = np.zeros((nf, len(points), nb), complex)
        if image_points:
            out["T_img"] = np.zeros_like(out["T"])
    if directions is not None:
        out["T_far"] = np.zeros((nf, len(directions), nb), complex)
    out["U_ap"] = np.zeros((nf, len(groups), nb), complex) if n_a else None
    band_of = np.concatenate([np.full(len(fb), j) for j, fb in enumerate(sub_bands)])
    band, band_id = None, -1
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f
        k = w / c
        if band_of[i] != band_id:
            fb = sub_bands[band_of[i]]
            band, band_id = bem.band(2 * np.pi * fb.min() / c, 2 * np.pi * fb.max() / c,
                                     ks_needed=2 * np.pi * np.asarray(fb) / c), band_of[i]
        P = band.solve(k)
        P_a, P_b = P[:, :n_a], P[:, n_a:]
        K = -1j * w * rho * (E_a @ P_a) if n_a else np.zeros((0, 0), complex)
        rhs = 1j * w * rho * (E_a @ (P_b @ G_bem)) if n_a and len(bx_tris) else np.zeros((n_a, nb), complex)
        N = np.zeros((n_a, nb), complex)
        Qbb = np.zeros((nb, nb), complex)
        for rows, AtD, A, cols, op in blocks:
            if len(cols):
                Qbb[np.ix_(cols, cols)] += op["Qbb"][i]
            if not len(rows):
                continue
            M = AtD @ op["Q"][i] @ AtD.T
            K[np.ix_(rows, rows)] += 1j * w * M
            if len(cols):
                N[np.ix_(rows, cols)] = AtD @ op["Qfb"][i]
        rhs = rhs - 1j * w * N
        V = sla.solve(K, rhs) if n_a else np.zeros((0, nb), complex)
        g = 1j * w * rho * np.vstack([V, G_bem])
        p_s = P @ g
        Za = -1j * w * (N.T @ V + Qbb)
        if len(bx_tris):
            Za += G_bem.T @ (E_b @ p_s)
        out["Za"][i] = Za
        if out["U_ap"] is not None:
            out["U_ap"][i] = Ug @ V
        if points is not None:
            Dm, Sm = bem.field_matrices(points, k)
            out["T"][i] = Dm @ p_s - Sm @ g
            if image_points:
                Dm, Sm = bem.field_matrices(points * [1, 1, -1], k)
                out["T_img"][i] = Dm @ p_s - Sm @ g
        if directions is not None:
            Dm, Sm = bem.field_matrices(directions, k, far=True)
            out["T_far"][i] = Dm @ p_s - Sm @ g
    for kk in ("Za", "T", "T_img", "T_far", "U_ap"):
        if out.get(kk) is not None:
            out[kk] = np.conj(out[kk])
    return out
