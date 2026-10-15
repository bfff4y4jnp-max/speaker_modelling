"""Split a project into independent solve jobs and merge the results.

Used by the GitHub Actions workflow (.github/workflows/solve.yml) and usable
on any batch system (Iridis/Slurm job arrays): every job is
    spkmod job solve REQUEST --part i --out part_i.npz
and the parts are combined with `spkmod job merge`.

A run request (runs/<name>.yml):
    project: examples/demo/project.yml
    scenes: [bass, tops]      # optional, default: all scenes
    per_octave: 12            # optional, overrides the project
    fmin: 20                  # optional
    fmax: 600                 # optional
    max_jobs: 20              # optional, parallel jobs (GitHub free: ~20)
    preset: party             # optional, DSP preset for the report
"""
from __future__ import annotations

import json
import os

import numpy as np
import yaml

from .scene import SceneResult, _bands


def load_request(path):
    with open(path) as fh:
        req = yaml.safe_load(fh) or {}
    req.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return req


def _project(req):
    from .project import Project
    pr = Project(req["project"])
    pr.settings.cache_dir = None
    return pr


def plan(req):
    """Deterministic list of jobs: [{part, scene, freqs}]."""
    pr = _project(req)
    scenes = req.get("scenes") or list(pr.cfg["scenes"])
    chunks = []
    for s in scenes:
        f = pr.freqs(s, req.get("fmin"), req.get("fmax"), req.get("per_octave"))
        for fb in _bands(f, pr.settings.band_octaves):
            # cost ~ (BEM size)^2 ~ f^4 above the h_max cap, plus a fixed setup cost
            h = min(pr.settings.h_max, pr.air.c / (fb.max() * pr.settings.elements_per_wavelength))
            cost = len(fb) * (pr.settings.h_max / h) ** 4 + 30
            chunks.append({"scene": s, "freqs": fb.tolist(), "cost": cost})
    max_jobs = int(req.get("max_jobs", 20))
    while len(chunks) > max_jobs:   # merge the cheapest adjacent pair within a scene
        pairs = [(chunks[i]["cost"] + chunks[i + 1]["cost"], i) for i in range(len(chunks) - 1)
                 if chunks[i]["scene"] == chunks[i + 1]["scene"]]
        if not pairs:
            break
        _, i = min(pairs)
        a, b = chunks[i], chunks.pop(i + 1)
        a["freqs"] += b["freqs"]
        a["cost"] += b["cost"]
    for i, c in enumerate(chunks):
        c["part"] = i
    return chunks


def solve_part(req, part, out):
    from .scene import solve_scene
    pr = _project(req)
    job = plan(req)[part]
    s = job["scene"]
    sc = pr.cfg["scenes"][s]
    ground = sc.get("ground", True)
    res = solve_scene(pr.cabinets(s), job["freqs"], points=pr.listeners(),
                      directions=pr.directions() if sc.get("polars", True) else None, ground=ground,
                      image_points=not ground, air=pr.air, settings=pr.settings, center=sc.get("acoustic_center"))
    res.meta.update({"scene": s, "part": part})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    res.save(out)
    return res


def merge(paths):
    """Concatenate part results of one scene along frequency."""
    parts = sorted((SceneResult.load(p) for p in paths), key=lambda r: r.freqs[0])
    r0 = parts[0]
    cat = lambda k: None if getattr(r0, k) is None else np.concatenate([getattr(p, k) for p in parts])
    U = None
    if r0.U_ap is not None:
        g = max(p.U_ap.shape[1] for p in parts)
        U = np.concatenate([np.pad(p.U_ap, ((0, 0), (0, g - p.U_ap.shape[1]), (0, 0))) for p in parts])
    meta = {"parts": [p.meta for p in parts], "time_s": sum(p.meta.get("time_s", 0) for p in parts)}
    return SceneResult(freqs=cat("freqs"), body_names=r0.body_names, Za=cat("Za"), T=cat("T"), T_img=cat("T_img"),
                       T_far=cat("T_far"), U_ap=U, ap_area=r0.ap_area, points=r0.points, directions=r0.directions,
                       center=r0.center, ground=r0.ground, meta=meta)


def merge_run(req, parts_dir, out_dir):
    """Merge every part in parts_dir into out_dir/results/<scene>.npz."""
    by_scene = {}
    for fn in sorted(os.listdir(parts_dir)):
        if fn.endswith(".npz"):
            r = SceneResult.load(os.path.join(parts_dir, fn))
            by_scene.setdefault(r.meta.get("scene"), []).append(os.path.join(parts_dir, fn))
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    info = {"request": req, "scenes": {}}
    for s, paths in by_scene.items():
        m = merge(paths)
        m.save(os.path.join(out_dir, "results", f"{s}.npz"), compact=True)
        info["scenes"][s] = {"freqs": [float(m.freqs[0]), float(m.freqs[-1]), len(m.freqs)],
                             "cpu_time_s": round(m.meta["time_s"]),
                             "meshes": sorted({x for p in m.meta["parts"] for x in p.get("meshes", [])})}
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return info
