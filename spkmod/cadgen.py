"""Panel-list YAML -> STEP (+ pistons snippet + preview PNG).

Lets a design discussion describe straight-cut plywood cabinets without a
CAD package. Spec format: see docs/HANDOVER.md. Units in the spec: mm.
"""
from __future__ import annotations

import os

import numpy as np
import yaml


def _solid(occ, p, base_dir):
    t = p["type"]
    if t == "box":
        tag = [occ.addBox(*p["at"], *p["size"])]
    elif t == "prism":
        plane, pts = p.get("plane", "xz"), p["points"]
        a0, a1 = p["extrude"]
        ax = {"xy": 2, "xz": 1, "yz": 0}[plane]
        uv = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
        P = []
        for q in pts:
            x = [0.0, 0.0, 0.0]
            x[uv[0]], x[uv[1]], x[ax] = q[0], q[1], a0
            P.append(occ.addPoint(*x))
        L = [occ.addLine(P[i], P[(i + 1) % len(P)]) for i in range(len(P))]
        s = occ.addPlaneSurface([occ.addCurveLoop(L)])
        d = [0.0, 0.0, 0.0]
        d[ax] = a1 - a0
        tag = [e[1] for e in occ.extrude([(2, s)], *d) if e[0] == 3]
    elif t in ("cylinder", "tube"):
        ax = np.asarray(p["axis"], float); ax /= np.linalg.norm(ax)
        L = p["length"]
        base = np.asarray(p["center"], float) - ax * L / 2
        if t == "cylinder":
            tag = [occ.addCylinder(*base, *(ax * L), p["diameter"] / 2)]
        else:
            o = occ.addCylinder(*base, *(ax * L), p["inner_d"] / 2 + p["wall"])
            i = occ.addCylinder(*base, *(ax * L), p["inner_d"] / 2)
            out, _ = occ.cut([(3, o)], [(3, i)])
            tag = [x[1] for x in out if x[0] == 3]
    elif t == "step":
        f = p["file"] if os.path.isabs(p["file"]) else os.path.join(base_dir, p["file"])
        tag = [x[1] for x in occ.importShapes(f) if x[0] == 3]
    else:
        raise ValueError(f"unknown part type {t!r}")
    r = p.get("rotate")
    if r:
        occ.rotate([(3, x) for x in tag], *r["about"], *r["axis"], np.radians(r["angle"]))
    return tag


def build(spec_path, out_dir=None, preview=True):
    """Build STEP from a panel spec file. Returns (step_path, pistons list in metres)."""
    with open(spec_path) as fh:
        spec = yaml.safe_load(fh)
    base = os.path.dirname(os.path.abspath(spec_path))
    return build_spec(spec, out_dir or base, base, preview=preview)


def build_spec(spec, out_dir, base_dir=".", preview=True, name=None, quiet=False):
    """Build STEP from a spec dict (same format as the YAML). Returns (step_path, pistons in metres)."""
    import gmsh
    base = base_dir
    os.makedirs(out_dir, exist_ok=True)
    name = name or spec["name"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        # gmsh options outlive initialize/finalize: the mesher sets metres, specs are in mm
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        occ = gmsh.model.occ
        solids = []
        for p in spec["parts"]:
            solids += _solid(occ, p, base)
        tools = []
        for c in spec.get("cutouts", []):
            cc = dict(c)
            if cc["type"] == "cylinder" and "length" not in cc:
                cc["length"] = cc.pop("depth", 200)
            tools += _solid(occ, cc, base)
        dts = [(3, s) for s in solids]
        if len(dts) > 1:
            dts, _ = occ.fuse(dts[:1], dts[1:])
        if tools:
            dts, _ = occ.cut(dts, [(3, t) for t in tools])
        occ.synchronize()
        step = os.path.join(out_dir, f"{name}.step")
        gmsh.write(step)
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        vol = sum(occ.getMass(3, t) for _, t in dts)
        if preview:
            gmsh.option.setNumber("Mesh.MeshSizeMax", max(bb[3:] - bb[:3]) / 25)
            gmsh.model.mesh.generate(2)
            tags, x, _ = gmsh.model.mesh.getNodes()
            idx = {int(t): i for i, t in enumerate(tags)}
            _, tn = gmsh.model.mesh.getElementsByType(2)
            tri = np.vectorize(idx.get)(tn.reshape(-1, 3))
            _preview(x.reshape(-1, 3), tri, spec, os.path.join(out_dir, f"{name}_preview.png"))
    finally:
        gmsh.finalize()
    pistons = []
    for p in spec.get("pistons", []):
        q = {"name": p["name"], "center": [round(v / 1000, 5) for v in p["center"]], "axis": p["axis"]}
        if "sd_cm2" in p:
            q["sd_cm2"] = p["sd_cm2"]
        if "cutout" in p:
            q["cutout"] = p["cutout"] / 1000
        if "thickness" in p:
            q["thickness"] = p["thickness"] / 1000
        if "motor" in p:
            q["motor"] = [v / 1000 for v in p["motor"]]
        for k in ("recess", "cone_depth", "basket_wall"):
            if k in p:
                q[k] = p[k] / 1000
        pistons.append(q)
    if quiet:
        return step, pistons
    print(f"{step}: bbox {bb[3] - bb[0]:.0f} x {bb[4] - bb[1]:.0f} x {bb[5] - bb[2]:.0f} mm, "
          f"wood volume {vol / 1e9 * 1000:.1f} L")
    print("cabinets:\n  %s:\n    step: %s\n    pistons:" % (name, os.path.relpath(step)))
    for q in pistons:
        print("      - " + yaml.safe_dump(q, default_flow_style=True, sort_keys=False, width=1000).strip())
    return step, pistons


def _preview(nodes, tri, spec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    seg = np.concatenate([nodes[tri[:, [0, 1]]], nodes[tri[:, [1, 2]]], nodes[tri[:, [2, 0]]]])
    for ax, (i, j, title) in zip(axs, [(0, 2, "side (x-z)"), (0, 1, "plan (x-y)"), (1, 2, "front (y-z)")]):
        ax.add_collection(LineCollection(seg[:, :, [i, j]], lw=0.2, colors="0.3"))
        for p in spec.get("pistons", []):
            c, a = np.asarray(p["center"]), np.asarray(p["axis"], float)
            ax.plot(c[i], c[j], "o", color="tab:orange")
            ax.arrow(c[i], c[j], 80 * a[i], 80 * a[j], color="tab:orange", width=4)
            ax.annotate(p["name"], (c[i], c[j]), color="tab:orange", fontsize=8)
        ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("xyz"[i] + " mm"); ax.set_ylabel("xyz"[j] + " mm")
    fig.suptitle(f"{spec['name']}: panels (grey), pistons and forward axis (orange)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
