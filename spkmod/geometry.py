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

import logging

import numpy as np

SOLID, AIR, PISTON = 0, 1, 2
KIND_RIGID, KIND_APERTURE, KIND_BODY = 0, 1, 2
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
def build_scene_mesh(cabinets, h, h_min=None, ground=True, piston_divisions=16, verbose=False, surface_only=False):
    """Build and mesh a scene. Returns SceneMesh.

    surface_only: mesh surfaces only (no air tetrahedra), for the reduced (reused-interior) solver.
    """
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
                c = np.asarray(p.center, float)
                base = c - ax * p.thickness / 2
                pis = occ.addCylinder(*base, *(ax * p.thickness), p.diameter / 2)
                local.append(((3, pis), PISTON, len(body_names)))
                cut = p.cutout or 1.2 * p.diameter
                fr = occ.addCylinder(*base, *(ax * p.thickness), cut / 2)
                local.append(((3, fr), SOLID, None))
                if p.motor:
                    md, mdep = p.motor
                    mo = occ.addCylinder(*(base), *(-ax * mdep), md / 2)
                    local.append(((3, mo), SOLID, None))
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
                    surf[s] = ("wall", None, None)
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
        solid_vols = [(3, t) for t, (cat, _, _) in cls.items() if cat != AIR]
        if solid_vols:
            occ.remove(solid_vols, recursive=False)
            occ.synchronize()
        gmsh.model.mesh.generate(2 if surface_only else 3)

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
