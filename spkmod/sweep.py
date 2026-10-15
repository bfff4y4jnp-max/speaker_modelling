"""Sweeps: many variants of a project, one comparison table.

    spkmod sweep sweep.yml [--parallel N] [--only a,b] [--dry-run]

sweep.yml (paths relative to the sweep file):

    name: sub_layouts                  # default: file name
    project: project.yml
    scenes: [stack]                    # scenes to include (default: all)
    out: sweeps/sub_layouts            # default: sweeps/<name> next to the sweep file
    frequencies: {min: 25, max: 165, per_octave: 24}   # optional, replaces the project's
    listeners: {file: floor_spots.csv} # optional, replaces the project's
    zones:                             # extra zones (the project's `zones:` and 'floor' are always included)
      dj: {box: [-0.5, 0.5, -1.5, -0.8], heights: [1.6], spacing: 0.25}
      behind: {box: [-8, -5, -3, 3], heights: [1.6], spacing: 0.5}
    bands: null                        # null: floor-model + sub bands; 'sixth'/'third'/'octave'; [[lo,hi],..]; {name: [lo,hi]}
    smoothing: 6                       # 1/n octave for spreads and holes
    dsp: project                       # project (CamillaDSP of the project) | flat (amp volts on every channel)
    reference: base                    # variant that d_median_db is taken against (default: the first)
    parallel: 1                        # solves at once; the memory budget and cores are split between them
    variants:
      - name: base
      - name: line4                    # geometry changes -> new solve (cached by content)
        move: {sub_rear_l: {position: [0, 1.1, 0], rotation: [0, 0, 180]}}
        remove: [sub_rear_r]
        add:
          - {cabinet: deep30, name: extra, position: [0, -1.1, 0], rotation: [0, 0, 180],
             wiring: {drv: {driver: 12-280-W, channel: sub_front}}}
        replace: {sub_front_l: {cabinet: fold3_35}}
      - name: gen                      # cabinet from a generator: f(**params) -> cadgen spec dict / YAML path
        cabinets: {deep30: {generator: make_specs.py:sub_from_params, params: {Dx: 1.18}}}
      - name: rear_delay               # offline only: reuses base's solve
        base: base
        grid: {offline.delay_ms.sub_rear: [3.0, 3.5, 4.0]}
        offline:
          polarity: {sub_rear: -1}     # also: gain_db, mute: [ch], preset: party, filters: {ch: [CamillaDSP filter dicts]}
      - name: finer
        project: {solver: {h_max: 0.08}}   # any deep merge into the project config

Outputs in `out`: table.csv, table.txt, fields/<key>.npz (per-channel pressure per volt, excursion per
volt, impedance) and variants/<variant>.npz (per-frequency zone medians, total pressure).
Solves are cached in the project's cache_dir by content, and each band group is saved as it finishes,
so an interrupted sweep resumes where it stopped.
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import logging
import multiprocessing as mp
import os
import time

import numpy as np
import yaml

log = logging.getLogger("spkmod")

GEOMETRY_KEYS = ("move", "remove", "add", "replace", "cabinets", "project", "wiring", "drivers")


# --------------------------------------------------------------------------- spec handling
def _deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_path(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand_variants(variants):
    """Resolve `base:` inheritance and `grid:` products. Returns [(name, variant dict)]."""
    byname = {}
    out = []
    for v in variants:
        v = dict(v)
        if "base" in v:
            if v["base"] not in byname:
                raise ValueError(f"variant {v.get('name')!r}: base {v['base']!r} must be defined earlier")
            v = _deep_merge(byname[v["base"]], {k: x for k, x in v.items() if k != "base"})
        name = v.get("name") or f"v{len(out)}"
        v["name"] = name
        grid = v.pop("grid", None)
        byname[name] = v
        if not grid:
            out.append((name, v))
            continue
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            g = copy.deepcopy(v)
            for k, val in zip(keys, combo):
                _set_path(g, k, val)
            label = ",".join(f"{k.split('.')[-1]}={val}" for k, val in zip(keys, combo))
            g["name"] = f"{name}[{label}]"
            out.append((g["name"], g))
    names = [n for n, _ in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate variant names: {sorted(dup)}")
    return out


def _load_generator(ref, base_dir):
    path, func = ref.rsplit(":", 1)
    path = path if os.path.isabs(path) else os.path.join(base_dir, path)
    spec = importlib.util.spec_from_file_location(f"spkmod_gen_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func)


def _generate_cabinet(name, gen, sweep_dir, out_dir, proj_dir):
    """Run a cabinet generator and build its STEP (cached by spec content)."""
    from .cadgen import build_spec
    fn = _load_generator(gen["generator"], sweep_dir)
    res = fn(**(gen.get("params") or {}))
    if isinstance(res, str):
        with open(res if os.path.isabs(res) else os.path.join(sweep_dir, res)) as fh:
            res = yaml.safe_load(fh)
    extra = {k: res.pop(k) for k in ("stuffing", "envelope") if k in res}
    h = hashlib.sha1(json.dumps(res, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cad_dir = os.path.join(out_dir, "cad")
    step = os.path.join(cad_dir, f"{name}_{h}.step")
    pist_file = step[:-5] + ".pistons.json"
    if not (os.path.exists(step) and os.path.exists(pist_file)):
        step_tmp, pistons = build_spec(res, cad_dir, os.path.dirname(os.path.join(sweep_dir, gen["generator"].rsplit(":", 1)[0])),
                                       preview=gen.get("preview", True), name=f"{name}_{h}", quiet=True)
        with open(pist_file, "w") as fh:
            json.dump(pistons, fh)
    with open(pist_file) as fh:
        pistons = json.load(fh)
    return {"step": os.path.abspath(step), "pistons": pistons, **extra}


def apply_variant(cfg, v, scenes, sweep_dir, out_dir, proj_dir):
    """Project config for one variant (does not touch the input)."""
    cfg = copy.deepcopy(cfg)
    if v.get("project"):
        cfg = _deep_merge(cfg, v["project"])
    for cab, c in (v.get("cabinets") or {}).items():
        if "generator" in c:
            cfg["cabinets"][cab] = _generate_cabinet(cab, c, sweep_dir, out_dir, proj_dir)
        else:
            cfg["cabinets"][cab] = _deep_merge(cfg["cabinets"].get(cab, {}), c)
    if v.get("drivers"):
        cfg["drivers"] = _deep_merge(cfg.get("drivers", {}), v["drivers"])
    cfg["scenes"] = {s: cfg["scenes"][s] for s in scenes}
    wiring = cfg.setdefault("wiring", {})

    def scene_of(item):
        for s in scenes:
            if any(it.get("name", it["cabinet"]) == item for it in cfg["scenes"][s]["items"]):
                return s
        raise KeyError(f"no item {item!r} in scenes {scenes}")

    for item, ch in (v.get("move") or {}).items():
        s = scene_of(item)
        for it in cfg["scenes"][s]["items"]:
            if it.get("name", it["cabinet"]) == item:
                it.update(ch)
    for item, ch in (v.get("replace") or {}).items():
        s = scene_of(item)
        for it in cfg["scenes"][s]["items"]:
            if it.get("name", it["cabinet"]) == item:
                it.update(ch)
    for item in v.get("remove") or []:
        s = scene_of(item)
        cfg["scenes"][s]["items"] = [it for it in cfg["scenes"][s]["items"] if it.get("name", it["cabinet"]) != item]
        wiring[s] = {b: e for b, e in (wiring.get(s) or {}).items() if not b.startswith(item + ".")}
    for it in v.get("add") or []:
        it = dict(it)
        s = it.pop("scene", scenes[0])
        w = it.pop("wiring", {}) or {}
        cfg["scenes"][s]["items"].append(it)
        wiring.setdefault(s, {}).update({f"{it['name']}.{p}": e for p, e in w.items()})
    for s, w in (v.get("wiring") or {}).items():
        wiring.setdefault(s, {}).update(w)
    for s in list(wiring):
        if s not in scenes:
            wiring.pop(s)
    return cfg


# --------------------------------------------------------------------------- solving
def _points(pr, spec, sweep_dir):
    """Observation points and zone index map: 'floor' = listeners, plus project and sweep zones."""
    from .metrics import zone_points
    pts, zones = [], {}
    L = pr.listeners() if spec.get("listeners") is None else None
    if spec.get("listeners") is not None:
        pr_l = copy.deepcopy(pr.cfg)
        pr_l["listeners"] = spec["listeners"]
        from .project import Project
        L = Project(cfg=pr_l, base_dir=sweep_dir).listeners()
    n = 0
    if L is not None:
        pts.append(L)
        zones["floor"] = np.arange(len(L))
        n = len(L)
    for src, base in ((pr.cfg.get("zones") or {}, pr.dir), (spec.get("zones") or {}, sweep_dir)):
        for z, zs in src.items():
            P = zone_points(zs, base)
            pts.append(P)
            zones[z] = np.arange(n, n + len(P))
            n += len(P)
    if not pts:
        raise ValueError("no listener points: give listeners or zones")
    return np.vstack(pts), zones


def _key(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _acoustic_signature(cfg, scenes, freqs, points):
    """Everything a scene solve depends on (wiring/drivers/DSP excluded)."""
    from .scene import _file_hash
    cabs = {}
    for s in scenes:
        for it in cfg["scenes"][s]["items"]:
            c = cfg["cabinets"][it["cabinet"]]
            cabs[it["cabinet"]] = {k: v for k, v in c.items() if k != "step"}
            cabs[it["cabinet"]]["step_hash"] = _file_hash(c["step"]) if c.get("step") and os.path.isabs(c["step"]) else c.get("step")
    sc = {s: {k: v for k, v in cfg["scenes"][s].items() if k not in ("ground_sigma", "ground_reflection")} for s in scenes}
    return {"cabs": cabs, "scenes": sc, "solver": {k: v for k, v in (cfg.get("solver") or {}).items()
                                                 if k not in ("memory_gb", "cache_dir")},
            "air": cfg.get("air"), "freqs": np.round(freqs, 4).tolist(), "points": np.round(points, 5).tolist()}


def _field_signature(cfg, scenes, acoustic):
    return {"a": acoustic, "wiring": {s: (cfg.get("wiring") or {}).get(s) for s in scenes},
            "drivers": cfg.get("drivers"), "ground": {s: [cfg["scenes"][s].get("ground_sigma"), cfg["scenes"][s].get("ground_reflection")] for s in scenes}}


def _solve_group(job):
    """Worker: solve the scenes once, then write the per-channel fields of every wiring variant."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(message)s")
    logging.getLogger("skfem").setLevel(logging.WARNING)
    logging.getLogger("pyopencl").setLevel(logging.WARNING)
    try:
        import bempp_cl.api as b
        b.set_logging_level("warning")
    except ImportError:
        pass
    from .project import Project
    from .scene import solve_scene
    t0 = time.time()
    results = None
    for cfg, base_dir, scenes, freqs, points, out_path in job["fields"]:
        pr = Project(cfg=cfg, base_dir=base_dir)
        if job.get("memory_gb"):
            pr.settings.memory_gb = job["memory_gb"]
        if results is None:
            results = {}
            for s in scenes:
                sc = pr.cfg["scenes"][s]
                ground = sc.get("ground", True)
                fs = freqs[(freqs >= (sc.get("min_freq") or 0) * 0.999) & (freqs <= (sc.get("max_freq") or 1e9) * 1.001)]
                results[s] = solve_scene(pr.cabinets(s), fs, points=points, directions=None, ground=ground,
                                         image_points=not ground, air=pr.air, settings=pr.settings,
                                         center=sc.get("acoustic_center"))
        save_fields(pr.system(results), out_path)
    return {"key": job["key"], "time_s": time.time() - t0}


def save_fields(sysm, path):
    from .scene import _atomic_savez
    ch = sysm.channel_responses()
    names = sorted(ch)
    bodies = [(c, b) for c in names for b in ch[c]["x"]]
    _atomic_savez(path, freqs=sysm.freqs, points=sysm.points, channels=np.array(names),
                  p=np.array([ch[c]["p"] for c in names]).astype(np.complex64),
                  Z=np.array([ch[c]["Z"] for c in names]).astype(np.complex64),
                  covered=np.array([ch[c]["covered"] for c in names]),
                  body_channel=np.array([c for c, _ in bodies]), body_name=np.array([b for _, b in bodies]),
                  x=np.array([ch[c]["x"][b] for c, b in bodies]).astype(np.complex64).reshape(len(bodies), -1))


def load_fields(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- offline processing
def channel_drive(pr, F, mode, off):
    """{channel: complex volts per unit program} including DSP mode and offline changes."""
    from .dsp import filter_response
    f = F["freqs"].astype(float)
    preset = off.get("preset")
    if mode != "flat" and not (pr.cfg.get("dsp") or {}).get("outputs"):
        mode = "flat"                       # no DSP in the project: drive every channel with the amp voltage
    if mode == "flat":
        amp = (pr.cfg.get("dsp") or {}).get("volts_at_0dbfs", 1.0)
        tf = {str(c): amp * np.ones(len(f), complex) for c in F["channels"]}
    else:
        tf = pr.channel_tf(f, preset)
    fs = 48000
    out = {}
    for c in map(str, F["channels"]):
        h = np.asarray(tf.get(c, np.zeros(len(f))), complex).copy()
        if c in (off.get("mute") or []):
            h[:] = 0
        h *= 10 ** ((off.get("gain_db") or {}).get(c, 0.0) / 20)
        h *= (off.get("polarity") or {}).get(c, 1.0)
        h *= np.exp(-2j * np.pi * f * (off.get("delay_ms") or {}).get(c, 0.0) / 1000)
        for flt in (off.get("filters") or {}).get(c, []):
            h *= filter_response(f, fs, flt, pr.dir)
        out[c] = h
    return out


def variant_response(pr, F, mode, off):
    drive = channel_drive(pr, F, mode, off)
    chans = list(map(str, F["channels"]))
    p = sum(drive[c][:, None] * F["p"][i].astype(complex) for i, c in enumerate(chans))
    x = {}
    for j, (c, b) in enumerate(zip(map(str, F["body_channel"]), map(str, F["body_name"]))):
        x[b] = x.get(b, 0) + drive[c] * F["x"][j].astype(complex)
    return p, x


# --------------------------------------------------------------------------- driver
def run_sweep(path, parallel=None, only=None, dry_run=False):
    from .metrics import format_table, parse_bands, zone_table
    from .project import Project
    sweep_dir = os.path.dirname(os.path.abspath(path))
    with open(path) as fh:
        spec = yaml.safe_load(fh)
    name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(sweep_dir, spec.get("out") or os.path.join("sweeps", name))
    os.makedirs(os.path.join(out_dir, "fields"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "variants"), exist_ok=True)
    proj_path = os.path.join(sweep_dir, spec["project"])
    with open(proj_path) as fh:
        base_cfg = yaml.safe_load(fh)
    proj_dir = os.path.dirname(os.path.abspath(proj_path))
    # absolute paths so variant configs work from anywhere
    for c in base_cfg.get("cabinets", {}).values():
        if c.get("step") and not os.path.isabs(c["step"]):
            c["step"] = os.path.join(proj_dir, c["step"])
    solver = base_cfg.setdefault("solver", {})
    solver["cache_dir"] = os.path.join(proj_dir, solver.get("cache_dir") or ".spkmod_cache")
    if spec.get("frequencies"):
        base_cfg["frequencies"] = spec["frequencies"]
    scenes = spec.get("scenes") or list(base_cfg["scenes"])
    base_pr = Project(cfg=base_cfg, base_dir=proj_dir)
    points, zones = _points(base_pr, spec, sweep_dir)
    fc = base_cfg.get("frequencies", {})
    from .system import log_freqs
    freqs = (np.asarray(spec["frequencies"]["list"], float) if isinstance(spec.get("frequencies"), dict)
             and "list" in spec["frequencies"] else log_freqs(fc.get("min", 20), fc.get("max", 1000), fc.get("per_octave", 24)))

    variants = expand_variants(spec["variants"])
    if only:
        keep = set(only)
        variants = [(n, v) for n, v in variants if n in keep or n.split("[")[0] in keep]
    plan, groups = [], {}
    for vname, v in variants:
        cfg = apply_variant(base_cfg, v, scenes, sweep_dir, out_dir, proj_dir)
        acoustic = _acoustic_signature(cfg, scenes, freqs, points)
        akey = _key(acoustic)
        fkey = _key(_field_signature(cfg, scenes, acoustic))
        fpath = os.path.join(out_dir, "fields", f"{fkey}.npz")
        plan.append((vname, v, cfg, fpath))
        if not os.path.exists(fpath):
            g = groups.setdefault(akey, {"key": akey, "fields": [], "variants": []})
            if fpath not in [f[-1] for f in g["fields"]]:
                g["fields"].append((cfg, proj_dir, scenes, freqs, points, fpath))
            g["variants"].append(vname)
    print(f"sweep {name}: {len(variants)} variants, {len(groups)} acoustic solves needed "
          f"({len(plan) - sum(len(g['variants']) for g in groups.values())} variants already have fields), "
          f"{len(points)} points, {len(freqs)} freqs, zones: {', '.join(f'{z}({len(i)})' for z, i in zones.items())}")
    for g in groups.values():
        print(f"  solve {g['key']}: {', '.join(g['variants'])}")
    if dry_run:
        return None

    n_par = max(1, int(parallel or spec.get("parallel", 1)))
    if groups:
        from .bem import available_memory
        mem = (solver.get("memory_gb") or 0.6 * available_memory() / 1e9) / n_par
        cores = os.cpu_count() or 1
        env = {k: str(max(1, cores // n_par)) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                                                         "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POCL_MAX_PTHREAD_COUNT")}
        jobs = [dict(g, memory_gb=mem) for g in groups.values()]
        t0 = time.time()
        failed = []
        if n_par == 1:
            for i, j in enumerate(jobs):
                print(f"[{i + 1}/{len(jobs)}] solving {j['key']} ({', '.join(j['variants'])})", flush=True)
                try:
                    r = _solve_group(j)
                    print(f"[{i + 1}/{len(jobs)}] done in {r['time_s']:.0f} s", flush=True)
                except Exception as exc:          # keep going; report at the end
                    log.exception("solve %s failed", j["key"])
                    failed.append((j, exc))
        else:
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                with cf.ProcessPoolExecutor(n_par, mp_context=mp.get_context("spawn")) as ex:
                    futs = {ex.submit(_solve_group, j): j for j in jobs}
                    for k, fut in enumerate(cf.as_completed(futs)):
                        j = futs[fut]
                        try:
                            r = fut.result()
                            print(f"[{k + 1}/{len(jobs)}] {j['key']} done in {r['time_s']:.0f} s "
                                  f"({time.time() - t0:.0f} s elapsed)", flush=True)
                        except Exception as exc:
                            log.error("solve %s failed: %r", j["key"], exc)
                            failed.append((j, exc))
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        if failed:
            print("FAILED solves (rerun the sweep to retry; finished bands are cached):")
            for j, exc in failed:
                print(f"  {', '.join(j['variants'])}: {exc!r}")

    # ---------------------------------------------------------------- tables
    mode = spec.get("dsp", "project")
    smoothing = spec.get("smoothing", 6)
    rows, medians = [], {}
    ref = spec.get("reference") or plan[0][0]
    bands = None
    for vname, v, cfg, fpath in plan:
        if not os.path.exists(fpath):
            continue
        F = load_fields(fpath)
        f = F["freqs"].astype(float)
        bands = bands or parse_bands(spec.get("bands"), f)
        pr = Project(cfg=cfg, base_dir=proj_dir)
        p, x = variant_response(pr, F, mode, v.get("offline") or {})
        vr = zone_table(f, p, zones, bands, smoothing)
        for r in vr:
            lo, hi = bands[r["band"]]
            m = (f >= lo) & (f <= hi)
            r["x_peak_mm"] = float(max((np.sqrt(2) * np.abs(xx[m]).max() * 1e3 for xx in x.values()), default=0.0))
            r["variant"] = vname
        rows += vr
        med = {z: 10 * np.log10(np.median(np.abs(p[:, idx]) ** 2, axis=1) / 4e-10) for z, idx in zones.items()}
        medians[vname] = med
        np.savez_compressed(os.path.join(out_dir, "variants", f"{_safe(vname)}.npz"), freqs=f,
                            zones=np.array(list(zones)), median_db=np.array([med[z] for z in zones]),
                            p=p.astype(np.complex64))
    refrow = {(r["zone"], r["band"]): r["median_db"] for r in rows if r["variant"] == ref}
    for r in rows:
        if (r["zone"], r["band"]) in refrow:
            r["d_median_db"] = r["median_db"] - refrow[(r["zone"], r["band"])]
    keys = ["variant", "zone", "band", "median_db", "d_median_db", "p10", "p90", "rms", "holes6_pct", "holes3_pct",
            "x_peak_mm", "n_freqs"]
    with open(os.path.join(out_dir, "table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    txt = (f"sweep {name}  (dsp: {mode}, reference: {ref}, levels dB SPL at 0 dBFS program, "
           f"spreads/holes at 1/{smoothing} oct)\n" + format_table(rows))
    with open(os.path.join(out_dir, "table.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"\nwritten: {os.path.join(out_dir, 'table.csv')}, table.txt, fields/, variants/")
    return rows


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_=.," else "_" for ch in s)
