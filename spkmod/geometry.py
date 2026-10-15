"""CAD -> tagged acoustic mesh (gmsh / OpenCASCADE).

A cabinet is described by its *solid* parts (wood, bracing, the driver
frame) from a STEP file or a builder function, plus pistons placed from the
project file. Everything inside the cabinet's envelope (bounding box by
default) that is not solid is air and is meshed with tetrahedra for the FEM.
The envelope surface is the FEM/BEM interface, so the exterior BEM only sees
a simple closed surface regardless of how intricate the folds are, and
recesses, open-backed panels or non-box shapes are handled exactly by the
FEM.

Surfaces are classified automatically:
  outer face of air      -> aperture (horn mouth, port, open back)
  outer face of solid    -> rigid exterior (BEM)
  outer face of piston   -> vibrating exterior face (BEM)
  air | solid            -> rigid cavity wall (FEM, with boundary-layer losses)
  air | piston           -> vibrating cavity face (FEM)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import dataclasses
import logging

import numpy as np

SOLID, AIR, PISTON = 0, 1, 2
KIND_RIGID, KIND_APERTURE, KIND_BODY = 0, 1, 2


class MeshError(RuntimeError):
    """gmsh could not mesh a scene; the message names the cabinet/piston involved where possible."""
log = logging.getLogger("spkmod")


@dataclass
class Piston:
    """Rigid diaphragm: a disc of area Sd, moving along `axis` (towards the front)."""
    name: str
    center: Sequence[float]
    axis: Sequence[float]
    diameter: float                      # effective (Sd = pi d^2/4)
    cutout: Optional[float] = None       # frame / baffle hole diameter, default 1.2 d
    thickness: float = 0.008
    motor: Optional[Sequence[float]] = None   # (diameter, depth) rigid motor behind the cone
    recess: float = 0.0                  # set the piston back by this much (cone dish air in front of it)
    cone_depth: Optional[float] = None   # alternative to recess: dish depth, volume Sd*depth/3 -> recess depth/3
    basket_wall: float = 0.005           # rigid basket ring sealing the dish from what is behind the baffle

    @property
    def recess_depth(self):
        return self.recess or ((self.cone_depth or 0.0) / 3)

    @property
    def sd(self):
        return np.pi * self.diameter ** 2 / 4


@dataclass
class Cabinet:
    name: str
    step: Optional[str] = None                     # STEP file of the solid parts (metres or mm, see units)
    builder: Optional[Callable] = None             # f(occ) -> list of volume tags (local coords)
    units: str = "mm"                              # STEP units
    pistons: list = field(default_factory=list)
    stuffing: list = field(default_factory=list)   # [(xmin,ymin,zmin,xmax,ymax,zmax), sigma]
    envelope: Optional[Sequence[float]] = None     # explicit box (xmin..zmax), default bbox
    position: Sequence[float] = (0.0, 0.0, 0.0)
    rotation: Sequence[float] = (0.0, 0.0, 0.0)    # degrees about x, y, z (applied in that order)


@dataclass
class SceneMesh:
    nodes: np.ndarray
    tets: np.ndarray
    tet_region: np.ndarray
    regions: dict                 # region id -> flow resistivity (or None)
    bem_tris: np.ndarray          # outward (into exterior fluid) oriented
    bem_kind: np.ndarray
    bem_body: np.ndarray          # body index or -1
    wall_tris: np.ndarray         # FEM rigid walls
    body_tris: np.ndarray         # FEM vibrating faces
    body_ids: np.ndarray
    body_names: list
    body_axes: np.ndarray
    body_centers: np.ndarray
    h: float
    body_area: np.ndarray = None  # true radiating area per body (one side)
    ap_group: np.ndarray = None   # per aperture triangle: CAD surface id (mouth / port grouping)
    ap_cab: np.ndarray = None     # per aperture triangle: index of the cabinet whose air it opens

    @property
    def aperture_tris(self):
        return self.bem_tris[self.bem_kind == KIND_APERTURE]

    def summary(self):
        return (f"{len(self.nodes)} nodes, {len(self.tets)} tets, BEM {len(self.bem_tris)} tris "
                f"({(self.bem_kind == KIND_APERTURE).sum()} aperture), {len(self.body_names)} bodies, h={self.h:.3f} m")


# --------------------------------------------------------------------- helpers
def hollow_box(outer, wall, holes=(), origin=(0, 0, 0)):
    """Builder for a rectangular box with round holes (driver cutouts, ports).

    holes: [(center, axis, diameter, depth)] cylinders subtracted from the walls.
    """
    def build(occ):
        x0, y0, z0 = origin
        W, H, D = outer
        o = occ.addBox(x0, y0, z0, W, H, D)
        i = occ.addBox(x0 + wall, y0 + wall, z0 + wall, W - 2 * wall, H - 2 * wall, D - 2 * wall)
        out, _ = occ.cut([(3, o)], [(3, i)])
        tools = []
        for c, ax, d, depth in holes:
            ax = np.asarray(ax, float) / np.linalg.norm(ax)
            p = np.asarray(c, float) - ax * depth / 2
            tools.append((3, occ.addCylinder(*p, *(ax * depth), d / 2)))
        if tools:
            out, _ = occ.cut(out, tools)
        return [t for d, t in out if d == 3]
    return build


def tube(center, axis, inner_d, length, wall):
    """Builder for a round port tube (solid wall), centred at `center`."""
    def build(occ):
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        p = np.asarray(center, float) - ax * length / 2
        o = occ.addCylinder(*p, *(ax * length), inner_d / 2 + wall)
        i = occ.addCylinder(*p, *(ax * length), inner_d / 2)
        out, _ = occ.cut([(3, o)], [(3, i)])
        return [t for d, t in out if d == 3]
    return build


def combine(*builders):
    def build(occ):
        tags = []
        for b in builders:
            tags += b(occ)
        out, _ = occ.fuse([(3, tags[0])], [(3, t) for t in tags[1:]]) if len(tags) > 1 else ([(3, tags[0])], None)
        return [t for d, t in out if d == 3]
    return build


def _rotmat(rot):
    ax, ay, az = np.radians(rot)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _orient_outward(nodes, tris):
    """Consistently orient a closed triangle surface (union of components) outward."""
    from collections import defaultdict, deque
    tris = tris.copy()
    edge_map = defaultdict(list)
    for i, t in enumerate(tris):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_map[(min(a, b), max(a, b))].append(i)
    seen = np.zeros(len(tris), bool)
    comps = []
    for s in range(len(tris)):
        if seen[s]:
            continue
        comp, q = [s], deque([s])
        seen[s] = True
        while q:
            i = q.popleft()
            t = tris[i]
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                for j in edge_map[(min(a, b), max(a, b))]:
                    if j == i or seen[j]:
                        continue
                    if len(edge_map[(min(a, b), max(a, b))]) != 2:
                        continue  # non-manifold edge: leave to the other side
                    u = tris[j]
                    # consistent if the shared edge runs opposite in the neighbour
                    same = any((u[k], u[(k + 1) % 3]) == (a, b) for k in range(3))
                    if same:
                        tris[j] = u[::-1]
                    seen[j] = True
                    comp.append(j)
                    q.append(j)
        comps.append(comp)
    for comp in comps:
        p = nodes[tris[comp]]
        vol = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()
        if vol < 0:
            tris[comp] = tris[comp][:, ::-1]
    return tris


# --------------------------------------------------------------------- build
NUDGES_MM = (0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0)


def build_scene_mesh(cabinets, h, h_min=None, ground=True, piston_divisions=16, verbose=False, surface_only=False,
                     auto_fix=True, refine_gaps=False):
    """Build and mesh a scene. Returns SceneMesh.

    surface_only: mesh surfaces only (no air tetrahedra), for the reduced (reused-interior) solver.
    refine_gaps: air gaps narrower than h/3 between facing CAD walls are always reported; True also
        meshes them finer (size max(3 gap, h/4)); None skips the check.
    auto_fix: if gmsh fails, find the piston whose parts (disc, frame ring, motor) make the
        geometry unmeshable - usually a face or edge that almost touches a CAD face - and nudge
        that driver along its axis by up to 3 mm (logged). Otherwise raise MeshError naming it.
    """
    kw = dict(h=h, h_min=h_min, ground=ground, piston_divisions=piston_divisions, verbose=verbose,
              surface_only=surface_only, refine_gaps=refine_gaps)
    try:
        return _build_scene_mesh(cabinets, **kw)
    except Exception as exc:
        first = _gmsh_msg(exc)
    names = [(ci, pi, f"{c.name}.{p.name}") for ci, c in enumerate(cabinets) for pi, p in enumerate(c.pistons)]
    culprits = []
    for ci, pi, name in names:                 # which driver's parts break the geometry?
        trial = list(cabinets)
        trial[ci] = dataclasses.replace(cabinets[ci], pistons=[p for k, p in enumerate(cabinets[ci].pistons) if k != pi])
        if _meshes(trial, kw):
            culprits.append((ci, pi, name))
    if not culprits:
        bad = [c.name for c in cabinets if len(cabinets) > 1 and not _meshes([c], kw)]
        raise MeshError(f"gmsh failed ({first}). It still fails with each piston removed, so the problem is in the "
                        f"CAD solids{' of ' + ', '.join(bad) if bad else ''}: look for faces that almost touch "
                        f"(gaps or steps under ~1 mm), tangent curved surfaces or thin slivers.")
    parts = {}
    for ci, pi, name in culprits:
        p = cabinets[ci].pistons[pi]
        what = []
        if p.motor and _meshes(_with_piston(cabinets, ci, pi, motor=None), kw):
            what.append("its motor (a motor face lies within ~1 mm of a CAD face)")
        if _meshes(_with_piston(cabinets, ci, pi, cutout=None), kw):
            what.append("its frame ring (the cutout circle coincides with or grazes a CAD edge/face)")
        parts[name] = what or ["its disc"]
    if auto_fix:
        fixed = list(cabinets)
        moved = []
        for ci, pi, name in culprits:
            p = fixed[ci].pistons[pi]
            ax = np.asarray(p.axis, float) / np.linalg.norm(p.axis)
            for d in NUDGES_MM:
                trial = _with_piston(fixed, ci, pi, center=tuple(np.asarray(p.center, float) + ax * d / 1000))
                if _meshes(trial, kw):
                    fixed, _ = trial, moved.append(f"{name} by {d:+.1f} mm along its axis")
                    break
        if len(moved) == len(culprits):
            log.warning("gmsh failed (%s) because of %s; meshed after moving %s. Move the piston in the "
                        "cabinet spec to make this permanent.", first,
                        "; ".join(f"{n}: {', '.join(w)}" for n, w in parts.items()), "; ".join(moved))
            return _build_scene_mesh(fixed, **kw)
    raise MeshError(f"gmsh failed ({first}) because of " + "; ".join(f"piston {n}: {', '.join(w)}" for n, w in parts.items())
                    + ". Move the piston a few mm along its axis, or change its cutout/motor so its faces and "
                    "edges do not touch the CAD faces" + (" (automatic nudges up to 3 mm did not help)" if auto_fix else ""))


def _gmsh_msg(exc):
    return str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__


def _meshes(cabinets, kw):
    try:
        _build_scene_mesh(cabinets, **{**kw, "refine_gaps": None})
        return True
    except Exception:
        return False


def _with_piston(cabinets, ci, pi, **change):
    out = list(cabinets)
    ps = list(out[ci].pistons)
    ps[pi] = dataclasses.replace(ps[pi], **change)
    out[ci] = dataclasses.replace(out[ci], pistons=ps)
    return out


def _find_gaps(gmsh, cls, surf, vol_cab, cabinets, h, ground):
    """Air gaps narrower than h/3 between two wall faces of the same scene (distance, faces, point)."""
    walls = [s for s, (where, _, pay) in surf.items() if where == "wall" and pay != "driver"]   # CAD walls only
    air = [t for t, (cat, _, _) in cls.items() if cat == AIR]
    bb = {s: np.array(gmsh.model.getBoundingBox(2, s)) for s in walls}
    curves = {s: {abs(c[1]) for c in gmsh.model.getBoundary([(2, s)], oriented=False)} for s in walls}
    lim = h / 3
    gaps = []
    for i, a in enumerate(walls):
        for b in walls[i + 1:]:
            A, B = bb[a], bb[b]
            if (np.maximum(A[:3] - B[3:], B[:3] - A[3:]) > lim).any() or curves[a] & curves[b]:
                continue
            d, *xyz = gmsh.model.occ.getDistance(2, a, 2, b)
            if not (1e-6 < d < lim):
                continue
            pa, pb = np.array(xyz[:3]), np.array(xyz[3:])
            if not _facing(gmsh, a, pa, b, pb):        # walls meeting at an angle (corners, wedges) are not gaps
                continue
            mid = (pa + pb) / 2
            owner = next((v for v in air if gmsh.model.isInside(3, v, list(mid))), None)
            # the whole segment between the walls must be air (not measured through a panel)
            if owner is not None and all(any(gmsh.model.isInside(3, v, list(pa + t * (pb - pa))) for v in air)
                                         for t in (0.1, 0.3, 0.7, 0.9)):
                gaps.append((d, a, b, mid, cabinets[vol_cab[owner]].name if owner in vol_cab else "?"))
    return gaps


def _facing(gmsh, a, pa, b, pb, cos_min=0.94):
    """True when the two faces are near-parallel at their closest points (within ~20 deg)."""
    try:
        na = np.array(gmsh.model.getNormal(a, gmsh.model.getParametrization(2, a, list(pa))))
        nb = np.array(gmsh.model.getNormal(b, gmsh.model.getParametrization(2, b, list(pb))))
    except Exception:
        return False
    na, nb = na / np.linalg.norm(na), nb / np.linalg.norm(nb)
    return abs(float(na @ nb)) > cos_min


def _build_scene_mesh(cabinets, h, h_min=None, ground=True, piston_divisions=16, verbose=False, surface_only=False,
                      refine_gaps=False):
    import gmsh
    h_min = h_min or h / 8
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        occ = gmsh.model.occ
        inputs = []          # (dimtag, category, payload)
        body_names, body_axes, body_centers, piston_r = [], [], [], []
        envelopes = []
        for cab in cabinets:
            R, T = _rotmat(cab.rotation), np.asarray(cab.position, float)
            local = []
            if cab.step:
                dts = occ.importShapes(cab.step)
                solids = [t for d, t in dts if d == 3]
            elif cab.builder:
                solids = cab.builder(occ)
            else:
                solids = []
            local += [((3, t), SOLID, None) for t in solids]
            for p in cab.pistons:
                ax = np.asarray(p.axis, float) / np.linalg.norm(p.axis)
                c0 = np.asarray(p.center, float)
                d = p.recess_depth
                c = c0 - ax * d                          # recessed piston (d = 0: in the baffle plane)
                base = c - ax * p.thickness / 2
                r_p = p.diameter / 2
                pis = occ.addCylinder(*base, *(ax * p.thickness), r_p)
                local.append(((3, pis), PISTON, len(body_names)))
                cut = p.cutout or 1.2 * p.diameter
                base0 = c0 - ax * p.thickness / 2
                if cut / 2 > r_p * (1 + 1e-6):           # frame ring in the baffle hole (no zero-width ring)
                    fr = occ.addCylinder(*base0, *(ax * p.thickness), cut / 2)
                    if d > 0:                            # leave the dish open to the front
                        hole = occ.addCylinder(*base0, *(ax * p.thickness), r_p)
                        fr = [t for dd, t in occ.cut([(3, fr)], [(3, hole)])[0] if dd == 3][0]
                    local.append(((3, fr), SOLID, "driver"))
                if d > 0:                                # basket wall: piston back plane -> frame back plane
                    ro = occ.addCylinder(*base, *(ax * d), r_p + p.basket_wall)
                    ri = occ.addCylinder(*base, *(ax * d), r_p)
                    for dd, t in occ.cut([(3, ro)], [(3, ri)])[0]:
                        if dd == 3:
                            local.append(((3, t), SOLID, "driver"))
                if p.motor:
                    md, mdep = p.motor
                    mo = occ.addCylinder(*(base), *(-ax * mdep), md / 2)
                    local.append(((3, mo), SOLID, "driver"))
                body_names.append(f"{cab.name}.{p.name}")
                body_axes.append(R @ ax)
                body_centers.append(R @ c + T)
                piston_r.append(p.diameter / 2)
            for box, sigma in cab.stuffing:
                b = occ.addBox(box[0], box[1], box[2], box[3] - box[0], box[4] - box[1], box[5] - box[2])
                local.append(((3, b), AIR, sigma))
            occ.synchronize()
            if cab.envelope is not None:
                e = cab.envelope
                env = occ.addBox(e[0], e[1], e[2], e[3] - e[0], e[4] - e[1], e[5] - e[2])
            else:
                bb = np.array([gmsh.model.getBoundingBox(*dt) for dt, cat, _ in local if cat != AIR])
                lo, hi = bb[:, :3].min(0), bb[:, 3:].max(0)
                env = occ.addBox(*lo, *(hi - lo))
            local.append(((3, env), AIR, None))
            dts = [dt for dt, _, _ in local]
            if cab.rotation != (0, 0, 0) and any(cab.rotation):
                for i, ang in enumerate(np.radians(cab.rotation)):
                    if ang:
                        axis = np.eye(3)[i]
                        occ.rotate(dts, 0, 0, 0, *axis, ang)
            occ.translate(dts, *T)
            envelopes.append(env)
            inputs += local
        occ.synchronize()

        dimtags = [dt for dt, _, _ in inputs]
        out, outmap = occ.fragment(dimtags, [])
        occ.synchronize()
        # classify fragment volumes by priority piston > solid > stuffing > air
        cls = {}
        for (dt, cat, payload), pieces in zip(inputs, outmap):
            for d, t in pieces:
                if d != 3:
                    continue
                prev = cls.get(t, (None, None, -1))
                rank = {PISTON: 3, SOLID: 2}.get(cat, 1 if payload is not None else 0)
                if rank > prev[2]:
                    cls[t] = (cat, payload, rank)
        env_pieces, vol_cab = set(), {}
        for (dt, cat, payload), pieces in zip(inputs, outmap):
            if dt[1] in envelopes:
                env_pieces |= {t for d, t in pieces if d == 3}
                for d, t in pieces:
                    if d == 3:
                        vol_cab[t] = envelopes.index(dt[1])
        outside = [t for t in cls if t not in env_pieces]
        if outside:
            occ.remove([(3, t) for t in outside], recursive=True)
            occ.synchronize()
            for t in outside:
                cls.pop(t)

        # surface classification
        surf, surf_cab = {}, {}
        for _, s in gmsh.model.getEntities(2):
            up, _ = gmsh.model.getAdjacencies(2, s)
            vols = [v for v in up if v in cls]
            if len(vols) == 1:
                cat, payload, _ = cls[vols[0]]
                surf[s] = ("outer", cat, payload)
                surf_cab[s] = vol_cab.get(vols[0], -1)
            elif len(vols) == 2:
                a, b = (cls[v] for v in vols)
                cats = {a[0], b[0]}
                if cats == {AIR, SOLID}:
                    surf[s] = ("wall", None, a[1] if a[0] == SOLID else b[1])     # payload 'driver' for frames/motors
                elif cats == {AIR, PISTON}:
                    pay = a[1] if a[0] == PISTON else b[1]
                    surf[s] = ("body", PISTON, pay)
                elif cats == {AIR} and vol_cab.get(vols[0]) != vol_cab.get(vols[1]):
                    log.warning("air of cabinets %s and %s touches: the reduced solver treats that face as rigid",
                                cabinets[vol_cab.get(vols[0])].name, cabinets[vol_cab.get(vols[1])].name)
        # mesh sizes
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        for s, (where, cat, payload) in surf.items():
            if cat == PISTON:
                r = piston_r[payload]
                pts = gmsh.model.getBoundary([(2, s)], recursive=True)
                gmsh.model.mesh.setSize(pts, min(h, 2 * np.pi * r / (2 * piston_divisions)))
        gap_note = gap_warn = None
        if refine_gaps is not None:
            gaps = _find_gaps(gmsh, cls, surf, vol_cab, cabinets, h, ground)
            if gaps and refine_gaps:
                smin, gaps_size = h, []
                for d, a, b, mid, cname in gaps:
                    size = min(h, max(1.5 * d, h_min))
                    smin = min(smin, size)
                    gaps_size.append(((a, b), size))
                fields = []
                for (a, b), size in gaps_size:
                    fid = gmsh.model.mesh.field.add("Constant")
                    gmsh.model.mesh.field.setNumbers(fid, "SurfacesList", [a, b])
                    gmsh.model.mesh.field.setNumber(fid, "VIn", size)
                    gmsh.model.mesh.field.setNumber(fid, "VOut", 1e22)
                    fields.append(fid)
                fmin = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
                seen = {}
                for d, a, b, mid, cname in sorted(gaps):
                    seen.setdefault((cname, round(d * 1000)), mid)
                gap_note = "; ".join(f"{c}: {g} mm near ({m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f})" for (c, g), m in seen.items())
            if gaps and not refine_gaps:
                seen = {}
                for d, a, b, mid, cname in sorted(gaps):
                    seen.setdefault((cname, round(d * 1000)), mid)
                gap_warn = "; ".join(f"{c}: {g} mm near ({m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f})" for (c, g), m in seen.items())
        solid_vols = [(3, t) for t, (cat, _, _) in cls.items() if cat != AIR]
        if solid_vols:
            occ.remove(solid_vols, recursive=False)
            occ.synchronize()
        gmsh.model.mesh.generate(2 if surface_only else 3)
        if gap_note:
            log.warning("narrow air gaps (< h/3 = %.0f mm) meshed finer: %s", h / 3 * 1000, gap_note)
        if gap_warn:
            log.warning("narrow air gaps (< h/3 = %.0f mm), meshed at h: %s. Set solver.refine_gaps: true to mesh "
                        "them finer, or check with a smaller h_max", h / 3 * 1000, gap_warn)

        tags, x, _ = gmsh.model.mesh.getNodes()
        idmap = np.zeros(int(tags.max()) + 1, np.int64)
        idmap[tags.astype(np.int64)] = np.arange(len(tags))
        nodes = x.reshape(-1, 3)
        tets, treg, regions = [], [], {}
        region_ids = {}
        for t, (cat, payload, _) in cls.items():
            if cat != AIR:
                continue
            _, tn = gmsh.model.mesh.getElementsByType(4, t)
            key = payload if payload else None
            rid = region_ids.setdefault(key, len(region_ids))
            regions[rid] = key
            tets.append(idmap[tn.astype(np.int64)].reshape(-1, 4))
            treg.append(np.full(len(tn) // 4, rid))
        groups = {k: [] for k in ("rigid", "aperture", "bodyext", "wall", "bodyint")}
        gbody = {"bodyext": [], "bodyint": []}
        ap_group, ap_cab = [], []
        for s, (where, cat, payload) in surf.items():
            _, tn = gmsh.model.mesh.getElementsByType(2, s)
            tri = idmap[tn.astype(np.int64)].reshape(-1, 3)
            if where == "outer":
                if cat == AIR:
                    on_ground = ground and np.all(np.abs(nodes[tri][:, :, 2]) < 1e-7)
                    groups["wall" if on_ground else "aperture"].append(tri)
                    if not on_ground:
                        ap_group.append(np.full(len(tri), s))
                        ap_cab.append(np.full(len(tri), surf_cab.get(s, -1)))
                    if on_ground:
                        groups["rigid"].append(tri)  # keeps the exterior surface closed (dropped by BEM)
                elif cat == PISTON:
                    groups["bodyext"].append(tri)
                    gbody["bodyext"].append(np.full(len(tri), payload))
                else:
                    groups["rigid"].append(tri)
            elif where == "wall":
                groups["wall"].append(tri)
            elif where == "body":
                groups["bodyint"].append(tri)
                gbody["bodyint"].append(np.full(len(tri), payload))
    finally:
        gmsh.finalize()

    cat_ = lambda L, w=3: np.vstack(L).astype(np.int64) if L else np.zeros((0, w), np.int64)
    rigid, aper, bext = cat_(groups["rigid"]), cat_(groups["aperture"]), cat_(groups["bodyext"])
    bem_tris = np.vstack([rigid, aper, bext])
    bem_kind = np.concatenate([np.full(len(rigid), KIND_RIGID), np.full(len(aper), KIND_APERTURE),
                               np.full(len(bext), KIND_BODY)])
    bem_body = np.concatenate([np.full(len(rigid) + len(aper), -1),
                               np.concatenate(gbody["bodyext"]) if gbody["bodyext"] else np.zeros(0, int)])
    if len(bem_tris):
        bem_tris = _orient_outward(nodes, bem_tris)
    tets_ = cat_(tets, 4)
    # drop nodes not used by anything
    used = np.unique(np.concatenate([tets_.ravel(), bem_tris.ravel()]))
    remap = -np.ones(len(nodes), np.int64)
    remap[used] = np.arange(len(used))
    R = lambda a: remap[a] if len(a) else a
    return SceneMesh(
        nodes=nodes[used], tets=R(tets_), tet_region=np.concatenate(treg) if treg else np.zeros(0, int),
        regions=regions, bem_tris=R(bem_tris), bem_kind=bem_kind, bem_body=bem_body,
        wall_tris=R(cat_(groups["wall"])), body_tris=R(cat_(groups["bodyint"])),
        body_ids=np.concatenate(gbody["bodyint"]) if gbody["bodyint"] else np.zeros(0, int),
        body_names=body_names, body_axes=np.array(body_axes).reshape(-1, 3),
        body_centers=np.array(body_centers).reshape(-1, 3), h=h,
        body_area=np.pi * np.asarray(piston_r) ** 2,
        ap_group=np.concatenate(ap_group) if ap_group else np.zeros(0, int),
        ap_cab=np.concatenate(ap_cab) if ap_cab else np.zeros(0, int))
