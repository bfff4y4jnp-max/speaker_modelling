"""Coupled FEM-BEM solve of a scene for unit body velocities.

For every frequency the scene is solved once per body DOF (multiple
right-hand sides on one factorisation). Results are independent of drivers,
wiring and DSP, so those can be changed afterwards at no acoustic cost.

Coupled system (e^{-iwt}), unknowns FEM pressure p and aperture velocity v_a
(piecewise constant, along the outward normal):
    S(w) p        - i w C_a v_a            = i w C_b u_out
    C_a^T p       - i w rho E P_a v_a      = i w rho E P_b v_b
where P is the BEM surface-pressure map per unit dp/dn and E integrates the
P1 surface pressure over aperture triangles. Pressure continuity is weak
(per aperture triangle), velocity continuity strong.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .air import Air
from .bem import ExteriorBEM, available_memory, memory_estimate
from .fem import InteriorFEM
from .geometry import KIND_APERTURE, KIND_BODY, build_scene_mesh

log = logging.getLogger("spkmod")


@dataclass
class SolverSettings:
    elements_per_wavelength: float = 6.0     # BEM/FEM mesh density at the top of each band
    h_max: float = 0.25                      # never coarser than this (geometry resolution)
    h_min: Optional[float] = None
    band_octaves: float = 1.0                # one mesh + interpolation set per band
    quad_order: int = 3
    losses: bool = True                      # viscothermal boundary-layer losses
    memory_gb: Optional[float] = None        # budget; default 60 % of available
    piston_divisions: int = 16
    cache_dir: Optional[str] = ".spkmod_cache"


@dataclass
class SceneResult:
    freqs: np.ndarray
    body_names: list
    Za: np.ndarray                    # (nf, nb, nb) radiation/load impedance, N s/m
    T: Optional[np.ndarray] = None    # (nf, npts, nb) pressure per unit body velocity
    T_img: Optional[np.ndarray] = None  # same at ground-mirrored points (impedance ground)
    T_far: Optional[np.ndarray] = None  # (nf, ndir, nb) far-field pattern  p ~ e^{-jkr}/r * T_far
    U_ap: Optional[np.ndarray] = None   # (nf, ngroups, nb) aperture volume velocity
    ap_area: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    directions: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    ground: bool = True
    meta: dict = field(default_factory=dict)

    def save(self, path):
        d = {k: v for k, v in self.__dict__.items() if v is not None and k not in ("meta", "body_names")}
        _atomic_savez(path, body_names=np.array(self.body_names), meta=json.dumps(self.meta), **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files if k not in ("meta", "body_names")}
        kw["ground"] = bool(kw.get("ground", True))
        return cls(body_names=[str(s) for s in z["body_names"]], meta=json.loads(str(z["meta"])), **kw)


def _bands(freqs, octaves):
    f = np.sort(np.asarray(freqs, float))
    out, start = [], 0
    for i in range(1, len(f) + 1):
        if i == len(f) or f[i] > f[start] * 2 ** octaves:
            out.append(f[start:i])
            start = i
    return out


def _tri_area_normal(nodes, tris):
    p = nodes[tris]
    cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    a = 0.5 * np.linalg.norm(cr, axis=1)
    return a, cr / (2 * a[:, None]), p.mean(1)


def _atomic_savez(path, **arrays):
    """Write .npz via a temporary file so an interrupted run never leaves a corrupt cache entry."""
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _save_part(path, part):
    arrays = {k: v for k, v in part.items() if isinstance(v, np.ndarray)}
    info = {k: v for k, v in part.items() if not isinstance(v, np.ndarray) and v is not None}
    _atomic_savez(path, info=json.dumps(info), **arrays)


def _load_part(path):
    z = np.load(path, allow_pickle=False)
    part = json.loads(str(z["info"]))
    for k in ("Za", "T", "T_img", "T_far", "U_ap", "ap_area", "center"):
        part[k] = z[k] if k in z.files else None
    return part


def _hash(obj):
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:16]


def solve_scene(cabinets, freqs, points=None, directions=None, ground=True, image_points=False,
                air: Air = Air(), settings: SolverSettings = SolverSettings(), center=None):
    """Solve a scene over `freqs`.

    points: (n,3) field points (listener grid, mic positions).
    directions: (n,3) unit vectors for far-field balloons/polars.
    ground: rigid ground plane at z=0 in the BEM (for boxes standing on it).
    image_points: also evaluate at points mirrored in z=0, for an impedance
        (grass) ground applied at system level to scenes solved without one.
    """
    t0 = time.time()
    freqs = np.unique(np.asarray(freqs, float))
    points = None if points is None else np.asarray(points, float)
    directions = None if directions is None else np.asarray(directions, float)
    key = None
    if settings.cache_dir:
        # STEP files by content (not path/mtime), settings without the run-environment fields
        cabs = [(c.name, os.path.basename(c.step) if c.step else None, _file_hash(c.step) if c.step else None,
                 repr(c.pistons), repr(c.stuffing), c.envelope, tuple(c.position), tuple(c.rotation)) for c in cabinets]
        st = {k: v for k, v in settings.__dict__.items() if k not in ("memory_gb", "cache_dir")}
        key = _hash((cabs, freqs.tolist(), None if points is None else points.round(6).tolist(),
                     None if directions is None else directions.round(6).tolist(), ground, image_points,
                     repr(air), sorted(st.items()), center, 4))
        path = os.path.join(settings.cache_dir, f"scene_{key}.npz")
        if os.path.exists(path):
            log.info("cache hit %s", path)
            return SceneResult.load(path)

    budget = (settings.memory_gb * 1e9) if settings.memory_gb else 0.6 * available_memory()
    rho, c = air.rho, air.c
    parts = []
    bands = _bands(freqs, settings.band_octaves)
    hs = [min(settings.h_max, c / (fb.max() * settings.elements_per_wavelength)) for fb in bands]
    groups = []                      # consecutive bands sharing one mesh
    for fb, h in zip(bands, hs):
        if groups and abs(groups[-1][0] - h) < 1e-9:
            groups[-1][1].append(fb)
        else:
            groups.append([h, [fb]])
    for gi, (h, sub) in enumerate(groups):
        fmax = max(fb.max() for fb in sub)
        part_path = os.path.join(settings.cache_dir, f"scene_{key}_part{gi}.npz") if key else None
        if part_path and os.path.exists(part_path):          # resume after an interruption
            log.info("resuming: band group %d from %s", gi, part_path)
            parts.append(_load_part(part_path))
            continue
        for attempt in range(4):
            mesh = build_scene_mesh(cabinets, h, settings.h_min, ground=ground,
                                    piston_divisions=settings.piston_divisions)
            n_act = int(((mesh.bem_kind == KIND_APERTURE) | (mesh.bem_kind == KIND_BODY)).sum())
            mult = 2 if ground else 1
            n_nodes_bem = len(np.unique(mesh.bem_tris))
            est = memory_estimate(n_nodes_bem, mult * n_nodes_bem, mult * n_act, 7)
            if est < budget:
                break
            h *= 1.25 * np.sqrt(est / budget)
            log.warning("mesh too large for memory budget (%.1f GB > %.1f GB): coarsening to h=%.3f m; "
                        "accuracy near %g Hz reduced", est / 1e9, budget / 1e9, h, fmax)
        log.info("bands up to %g Hz: %s", fmax, mesh.summary())
        parts.append(_solve_band(mesh, sub, points, directions, ground, image_points, air, settings))
        if part_path:
            os.makedirs(settings.cache_dir, exist_ok=True)
            _save_part(part_path, parts[-1])
    res = SceneResult(
        freqs=freqs, body_names=parts[0]["names"],
        Za=np.concatenate([p["Za"] for p in parts]),
        T=np.concatenate([p["T"] for p in parts]) if points is not None else None,
        T_img=np.concatenate([p["T_img"] for p in parts]) if image_points and points is not None else None,
        T_far=np.concatenate([p["T_far"] for p in parts]) if directions is not None else None,
        U_ap=_stack_groups([p["U_ap"] for p in parts]),
        ap_area=parts[-1]["ap_area"],
        points=points, directions=directions,
        center=np.asarray(center if center is not None else parts[0]["center"], float),
        ground=ground,
        meta={"time_s": time.time() - t0, "meshes": [p["mesh"] for p in parts]})
    if key:
        os.makedirs(settings.cache_dir, exist_ok=True)
        res.save(os.path.join(settings.cache_dir, f"scene_{key}.npz"))
        for gi in range(len(groups)):
            pp = os.path.join(settings.cache_dir, f"scene_{key}_part{gi}.npz")
            if os.path.exists(pp):
                os.remove(pp)
    return res


def _stack_groups(arrs):
    if not arrs or arrs[0] is None:
        return None
    g = max(a.shape[1] for a in arrs)
    return np.concatenate([np.pad(a, ((0, 0), (0, g - a.shape[1]), (0, 0))) for a in arrs])


def _solve_band(mesh, sub_bands, points, directions, ground, image_points, air, settings):
    rho, c = air.rho, air.c
    freqs = np.concatenate(sub_bands)
    nodes = mesh.nodes
    nb = len(mesh.body_names)
    ap = mesh.bem_kind == KIND_APERTURE
    bx = mesh.bem_kind == KIND_BODY
    ap_tris, bx_tris = mesh.bem_tris[ap], mesh.bem_tris[bx]
    active = np.concatenate([np.where(ap)[0], np.where(bx)[0]])
    n_a = len(ap_tris)

    # body velocity shapes s_in (velocity into the fluid per unit DOF velocity)
    def shapes(tris, ids, outward_normals=None):
        G = np.zeros((len(tris), nb))
        if not len(tris):
            return G
        _, nrm, cen = _tri_area_normal(nodes, tris)
        if outward_normals is None:   # orient away from the body centre
            sgn = np.sign(np.einsum("ij,ij->i", nrm, cen - mesh.body_centers[ids]))
            nrm = nrm * sgn[:, None]
        G[np.arange(len(tris)), ids] = np.einsum("ij,ij->i", nrm, mesh.body_axes[ids])
        return G

    G_fem = shapes(mesh.body_tris, mesh.body_ids)
    G_bem = shapes(bx_tris, mesh.bem_body[bx], outward_normals=True)
    # exact volume velocity: rescale for the polygonal discretisation of round diaphragms
    if nb and mesh.body_area is not None:
        a_f = _tri_area_normal(nodes, mesh.body_tris)[0] if len(mesh.body_tris) else np.zeros(0)
        a_b = _tri_area_normal(nodes, bx_tris)[0] if len(bx_tris) else np.zeros(0)
        meshed = (np.abs(G_fem) * a_f[:, None]).sum(0) + (np.abs(G_bem) * a_b[:, None]).sum(0)
        scale = np.where(meshed > 0, 2 * mesh.body_area / np.maximum(meshed, 1e-30), 1.0)
        G_fem, G_bem = G_fem * scale, G_bem * scale

    bem = ExteriorBEM(nodes, mesh.bem_tris, active, ground=0.0 if ground else None, quad_order=settings.quad_order)
    red = np.full(len(nodes), -1)
    red[bem.used] = np.arange(bem.n)

    def E_matrix(tris):
        a, _, _ = _tri_area_normal(nodes, tris)
        r = np.repeat(np.arange(len(tris)), 3)
        return sp.csr_matrix((np.repeat(a / 3, 3), (r, red[tris].ravel())), shape=(len(tris), bem.n))

    E_a, E_b = E_matrix(ap_tris), E_matrix(bx_tris)
    has_fem = len(mesh.tets) > 0
    if has_fem:
        fem = InteriorFEM(nodes, mesh.tets, mesh.tet_region, mesh.regions, mesh.wall_tris, air, settings.losses)
        C_a = fem.facet_matrix(ap_tris)
        C_b = fem.facet_matrix(mesh.body_tris) if len(mesh.body_tris) else sp.csc_matrix((fem.ndof, 0))
    groups = np.unique(mesh.ap_group) if n_a else np.zeros(0, int)
    a_ap, _, _ = _tri_area_normal(nodes, ap_tris) if n_a else (np.zeros(0), None, None)
    Ug = sp.csr_matrix((a_ap, (np.searchsorted(groups, mesh.ap_group), np.arange(n_a))),
                       shape=(len(groups), n_a)) if n_a else None

    band_of = np.concatenate([np.full(len(fb), j) for j, fb in enumerate(sub_bands)])
    band, band_id = None, -1
    nf = len(freqs)
    out = {"names": mesh.body_names, "Za": np.zeros((nf, nb, nb), complex),
           "center": nodes[np.unique(mesh.bem_tris)].mean(0) if len(mesh.bem_tris) else np.zeros(3),
           "mesh": mesh.summary(), "ap_area": np.asarray(Ug.sum(1)).ravel() if Ug is not None else None}
    if points is not None:
        out["T"] = np.zeros((nf, len(points), nb), complex)
        if image_points:
            out["T_img"] = np.zeros_like(out["T"])
    if directions is not None:
        out["T_far"] = np.zeros((nf, len(directions), nb), complex)
    out["U_ap"] = np.zeros((nf, len(groups), nb), complex) if n_a else None

    for i, f in enumerate(freqs):
        w = 2 * np.pi * f
        k = w / c
        if band_of[i] != band_id:          # frequency-interpolation set for this sub-band
            fb = sub_bands[band_of[i]]
            band, band_id = bem.band(2 * np.pi * fb.min() / c, 2 * np.pi * fb.max() / c), band_of[i]
        P = band.solve(k)                       # (n, n_active) per unit dp/dn
        P_a, P_b = P[:, :n_a], P[:, n_a:]
        if has_fem and n_a:
            S = fem.operator(f)
            Dense = -1j * w * rho * (E_a @ P_a)
            top = sp.hstack([S, -1j * w * C_a])
            bot = sp.hstack([sp.csr_matrix(C_a.T), sp.csr_matrix(Dense)])
            Amat = sp.vstack([top, bot]).tocsc()
            rhs_top = 1j * w * (C_b @ -G_fem) if C_b.shape[1] else np.zeros((fem.ndof, nb))
            rhs_bot = 1j * w * rho * (E_b @ (P_b @ G_bem)) if len(bx_tris) else np.zeros((n_a, nb))
            sol = spla.splu(Amat).solve(np.vstack([rhs_top, rhs_bot]).astype(complex))
            p_fem, v_a = sol[:fem.ndof], sol[fem.ndof:]
        elif has_fem:   # closed cavities only (no aperture)
            p_fem = spla.splu(fem.operator(f)).solve((1j * w * (C_b @ -G_fem)).astype(complex))
            v_a = np.zeros((0, nb))
        else:
            p_fem, v_a = None, np.zeros((0, nb))
        g = 1j * w * rho * np.vstack([v_a, G_bem])            # dp/dn on active triangles
        p_s = P @ g
        Za = np.zeros((nb, nb), complex)
        if p_fem is not None and C_b.shape[1]:
            Za += G_fem.T @ (C_b.T @ p_fem)
        if len(bx_tris):
            Za += G_bem.T @ (E_b @ p_s)
        out["Za"][i] = Za
        if out["U_ap"] is not None:
            out["U_ap"][i] = Ug @ v_a
        if points is not None:
            Dm, Sm = bem.field_matrices(points, k)
            out["T"][i] = Dm @ p_s - Sm @ g
            if image_points:
                Dm, Sm = bem.field_matrices(points * [1, 1, -1], k)
                out["T_img"][i] = Dm @ p_s - Sm @ g
        if directions is not None:
            Dm, Sm = bem.field_matrices(directions, k, far=True)
            out["T_far"][i] = Dm @ p_s - Sm @ g
    # to engineering convention e^{+jwt}
    for kk in ("Za", "T", "T_img", "T_far", "U_ap"):
        if out.get(kk) is not None:
            out[kk] = np.conj(out[kk])
    return out
