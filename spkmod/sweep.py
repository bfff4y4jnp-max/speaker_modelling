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
    reference_csv: ../floor_model_deep30_x80.csv   # optional: floor zone vs the floor model, per band
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

Outputs in `out`: table.csv, table.txt, perfreq.csv (zone median and spread per frequency),
reference.csv (when reference_csv is given), fields/<key>.npz (per-channel pressure per volt, excursion per
volt, impedance) and variants/<variant>.npz (per-frequency zone medians, total pressure).
Sweeps default to solver.interior: reuse (each cabinet's interior solved once; set `full` in the
project or a variant's `project:` to force the coupled solve). Solves are cached in the project's cache_dir by content, and each band group is saved as it finishes,
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
    return {"step": os.path.abspath(step), "pistons": pistons, "spec_hash": h, **extra}


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
    """Observation points and zone index map: 'floor' = listeners, then project zones, then sweep zones."""
    from .metrics import zone_points
    from .project import Project
    if spec.get("listeners") is not None:
        cfg = copy.deepcopy(pr.cfg)
        cfg["listeners"] = spec["listeners"]
        pr = Project(cfg=cfg, base_dir=sweep_dir)
    pts, zones = pr.observation_points()
    pts = [] if pts is None else [pts]
    n = sum(len(p) for p in pts)
    for z, zs in (spec.get("zones") or {}).items():
        P = zone_points(zs, sweep_dir)
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
            if "spec_hash" not in c:        # generated cabinets are keyed by their spec, not the STEP bytes
                cabs[it["cabinet"]]["step_hash"] = (_file_hash(c["step"]) if c.get("step") and os.path.isabs(c["step"])
                                                    else c.get("step"))
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
                if not len(fs):             # a frequency shard outside this scene's range
                    continue
                results[s] = solve_scene(pr.cabinets(s), fs, points=points, directions=None, ground=ground,
                                         image_points=not ground, air=pr.air, settings=pr.scene_settings(s),
                                         center=sc.get("acoustic_center"))
            if job.get("acoustic_out"):            # scene results before drivers (monte_carlo reuses them)
                for s, r in results.items():
                    path = job["acoustic_out"].format(scene=s)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    r.save(path, compact=True)
        save_fields(pr.system(results), out_path)
    return {"key": job["key"], "time_s": time.time() - t0}


def save_fields(sysm, path):
    from .scene import _atomic_savez
    _atomic_savez(path, **fields_from_system(sysm))


def fields_from_system(sysm):
    """Per-channel fields (the fields/*.npz content) of a SystemModel, in memory."""
    ch = sysm.channel_responses()
    names = sorted(ch)
    bodies = [(c, b) for c in names for b in ch[c]["x"]]
    return dict(freqs=sysm.freqs, points=sysm.points, channels=np.array(names),
                p=np.array([ch[c]["p"] for c in names]).astype(np.complex64),
                p_img=np.array([ch[c]["p_img"] for c in names]).astype(np.complex64),
                center=np.array([ch[c]["center"] for c in names]).reshape(len(names), 3),
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
    crowd = off.get("crowd")
    if crowd:
        if "center" not in F:
            raise ValueError("offline crowd needs fields written by spkmod >= brief-2 item 4: delete the sweep's "
                             "fields/ and re-run")
        from .system import crowd_factors
        f = F["freqs"].astype(float)
        p = 0
        for i, c in enumerate(chans):
            Ad, Ai = crowd_factors(f, F["center"][i], F["points"], crowd, pr.air)
            pi = F["p_img"][i].astype(complex)
            p = p + drive[c][:, None] * ((F["p"][i].astype(complex) - pi) * Ad + pi * Ai)
    else:
        p = sum(drive[c][:, None] * F["p"][i].astype(complex) for i, c in enumerate(chans))
    x = {}
    for j, (c, b) in enumerate(zip(map(str, F["body_channel"]), map(str, F["body_name"]))):
        x[b] = x.get(b, 0) + drive[c] * F["x"][j].astype(complex)
    return p, x


# --------------------------------------------------------------------------- driver
def prepare(path, only=None, out=None, quiet=False):
    """Resolve a sweep file: variants, their project configs, solve groups, points, zones, frequencies."""
    from .project import Project
    sweep_dir = os.path.dirname(os.path.abspath(path))
    with open(path) as fh:
        spec = yaml.safe_load(fh)
    name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.abspath(out) if out else os.path.join(sweep_dir, spec.get("out") or os.path.join("sweeps", name))
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
    solver.setdefault("interior", "reuse")      # layouts change between variants: solve each cabinet's interior once
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
        v = dict(v, _akey=akey)
        if v.get("monte_carlo"):
            _check_mc(v["monte_carlo"], vname)
        plan.append((vname, v, cfg, fpath))
        need_ac = bool(v.get("monte_carlo")) and not os.path.exists(os.path.join(out_dir, "acoustic", akey, "done"))
        if not os.path.exists(fpath) or need_ac:
            g = groups.setdefault(akey, {"key": akey, "fields": [], "variants": [], "acoustic": False})
            if fpath not in [f[-1] for f in g["fields"]]:
                g["fields"].append((cfg, proj_dir, scenes, freqs, points, fpath))
            g["variants"].append(vname)
            g["acoustic"] = g["acoustic"] or need_ac
    ctx = dict(spec=spec, name=name, sweep_dir=sweep_dir, out_dir=out_dir, proj_dir=proj_dir, solver=solver,
               scenes=scenes, points=points, zones=zones, freqs=freqs, plan=plan, groups=groups, n_variants=len(variants),
               band_octaves=base_pr.settings.band_octaves, h_max=base_pr.settings.h_max,
               epw=base_pr.settings.elements_per_wavelength, c=base_pr.air.c)
    if quiet:
        return ctx
    print(f"sweep {name}: {len(variants)} variants, {len(groups)} acoustic solves needed "
          f"({len(plan) - sum(len(g['variants']) for g in groups.values())} variants already have fields), "
          f"{len(points)} points, {len(freqs)} freqs, zones: {', '.join(f'{z}({len(i)})' for z, i in zones.items())}")
    for g in groups.values():
        print(f"  solve {g['key']}: {', '.join(g['variants'])}")
    return ctx


def run_sweep(path, parallel=None, only=None, dry_run=False, out=None):
    ctx = prepare(path, only, out)
    if dry_run:
        return None
    spec, solver, groups = ctx["spec"], ctx["solver"], ctx["groups"]

    n_par = max(1, int(parallel or spec.get("parallel", 1)))
    if groups:
        from .bem import available_memory
        mem = (solver.get("memory_gb") or 0.6 * available_memory() / 1e9) / n_par
        cores = os.cpu_count() or 1
        env = {k: str(max(1, cores // n_par)) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                                                         "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POCL_MAX_PTHREAD_COUNT")}
        jobs = [dict(g, memory_gb=mem,
                     acoustic_out=(os.path.join(ctx["out_dir"], "acoustic", g["key"], "{scene}.npz") if g.get("acoustic")
                                   else None)) for g in groups.values()]
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

    for g in groups.values():
        d = os.path.join(ctx["out_dir"], "acoustic", g["key"])
        if g.get("acoustic") and os.path.isdir(d):
            open(os.path.join(d, "done"), "w").close()
    return tables(ctx)


def tables(ctx):
    """Comparison tables from the saved fields (no solving)."""
    from .metrics import compare_to_reference, format_table, parse_bands, per_frequency, zone_table
    from .project import Project
    spec, name, out_dir, sweep_dir, proj_dir = ctx["spec"], ctx["name"], ctx["out_dir"], ctx["sweep_dir"], ctx["proj_dir"]
    plan, points, zones = ctx["plan"], ctx["points"], ctx["zones"]
    # ---------------------------------------------------------------- tables
    mode = spec.get("dsp", "project")
    smoothing = spec.get("smoothing", 6)
    rows, medians, perfreq, refrows = [], {}, [], []
    ref_csv = spec.get("reference_csv")
    ref_csv = os.path.join(sweep_dir, ref_csv) if ref_csv and not os.path.isabs(ref_csv) else ref_csv
    ref = spec.get("reference") or plan[0][0]
    bands = None
    missing = [vname for vname, v, cfg, fpath in plan if not os.path.exists(fpath)]
    for vname, v, cfg, fpath in plan:
        if not os.path.exists(fpath):
            continue
        F = load_fields(fpath)
        f = F["freqs"].astype(float)
        bands = bands or parse_bands(spec.get("bands"), f)
        pr = Project(cfg=cfg, base_dir=proj_dir)
        p, x = variant_response(pr, F, mode, v.get("offline") or {})
        vr = _rows(f, p, x, zones, bands, smoothing, vname)
        rows += vr
        if v.get("monte_carlo"):
            rows += monte_carlo_rows(ctx, vname, v, cfg, mode, zones, bands, smoothing)
        med = {z: 10 * np.log10(np.median(np.abs(p[:, idx]) ** 2, axis=1) / 4e-10) for z, idx in zones.items()}
        medians[vname] = med
        perfreq += [{"variant": vname, **r} for r in per_frequency(f, p, zones, smoothing)]
        if ref_csv:
            cmp_ = compare_to_reference(f, p[:, zones["floor"]], points[zones["floor"]], ref_csv, bands)
            refrows += [{"variant": vname, "offset_db": cmp_["offset_db"], **r} for r in cmp_["rows"]]
        np.savez_compressed(os.path.join(out_dir, "variants", f"{_safe(vname)}.npz"), freqs=f, points=points,
                            zones=np.array(list(zones)), zone_index=np.concatenate(list(zones.values())),
                            zone_len=np.array([len(i) for i in zones.values()]),
                            median_db=np.array([med[z] for z in zones]), p=p.astype(np.complex64))
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
    _write_csv(os.path.join(out_dir, "perfreq.csv"), perfreq, ["variant", "zone", "freq", "median_db", "p10", "p90"])
    extra = ""
    if refrows:
        rk = ["variant", "band", "offset_db", "d_median_db", "spot_rms_db", "model_p10", "ref_p10", "model_p90",
              "ref_p90", "model_holes6_pct", "ref_holes6_pct", "model_holes3_pct", "ref_holes3_pct"]
        _write_csv(os.path.join(out_dir, "reference.csv"), refrows, rk)
        extra = "\n\n" + format_reference(refrows, os.path.basename(ref_csv))
    if missing:
        extra = (f"\n\n!! INCOMPLETE: no solved fields for {len(missing)} of {len(plan)} variants: "
                 + ", ".join(missing) + extra)
    txt = (f"sweep {name}  (dsp: {mode}, reference: {ref}, levels dB SPL at 0 dBFS program, "
           f"spreads/holes at 1/{smoothing} oct)\n" + format_table(rows) + extra)
    with open(os.path.join(out_dir, "table.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"\nwritten: {os.path.join(out_dir, 'table.csv')}, table.txt, fields/, variants/")
    if missing:
        raise SweepIncomplete(f"no solved fields for {len(missing)} of {len(plan)} variants: {', '.join(missing)}")
    return rows


class SweepIncomplete(RuntimeError):
    """Some variants have no solved fields: the table was written without them."""


# --------------------------------------------------------------------------- monte carlo
_MC_OK = ("drivers.", "wiring.", "dsp.")
MC_METRICS = ("median_db", "p10", "p90", "rms", "holes6_pct", "holes3_pct", "x_peak_mm")


def _check_mc(mc, vname):
    for path, d in (mc.get("vary") or {}).items():
        parts = path.split(".")
        ok = path.startswith(_MC_OK) or (parts[0] == "scenes" and len(parts) == 3
                                         and parts[2] == "ground_sigma")
        if not ok:
            raise ValueError(f"variant {vname!r}: monte_carlo can only vary drivers.*, wiring.*, dsp.* and "
                             f"scenes.<scene>.ground_sigma (offline inputs), not {path!r}")
        if not ({"rel", "sd_rel", "sd", "range"} & set(d)):
            raise ValueError(f"variant {vname!r}: monte_carlo vary {path!r} needs rel, sd_rel, sd or range")


def _get_path(cfg, path):
    d = cfg
    for k in path.split("."):
        d = d[k]
    return d


def mc_samples(mc, cfg):
    """List of (dotted path -> value) dicts, Latin hypercube (default) or plain random."""
    from scipy.stats import norm
    vary = mc["vary"]
    n = int(mc.get("n", 64))
    rng = np.random.default_rng(mc.get("seed", 1))
    keys = list(vary)
    U = np.empty((n, len(keys)))
    for j in range(len(keys)):
        U[:, j] = ((rng.permutation(n) + rng.random(n)) / n) if mc.get("method", "lhs") == "lhs" else rng.random(n)
    out = []
    for i in range(n):
        s = {}
        for j, key in enumerate(keys):
            d, u = vary[key], U[i, j]
            if "range" in d:
                lo, hi = d["range"]
                s[key] = lo + (hi - lo) * u
                continue
            nom = float(_get_path(cfg, key))
            if "rel" in d:
                s[key] = nom * (1 + d["rel"] * (2 * u - 1))
            elif "sd_rel" in d:
                s[key] = nom * (1 + d["sd_rel"] * norm.ppf(u))
            else:
                s[key] = nom + d["sd"] * norm.ppf(u)
        out.append(s)
    return out


def _rows(f, p, x, zones, bands, smoothing, vname):
    from .metrics import zone_table
    vr = zone_table(f, p, zones, bands, smoothing)
    for r in vr:
        lo, hi = bands[r["band"]]
        m = (f >= lo) & (f <= hi)
        r["x_peak_mm"] = float(max((np.sqrt(2) * np.abs(xx[m]).max() * 1e3 for xx in x.values()), default=0.0))
        r["variant"] = vname
    return vr


def monte_carlo_rows(ctx, vname, v, cfg, mode, zones, bands, smoothing):
    """p5 / p50 / p95 rows over offline-only perturbations, reusing the variant's stored scene results."""
    from .project import Project
    from .scene import SceneResult
    mc = v["monte_carlo"]
    d = os.path.join(ctx["out_dir"], "acoustic", v["_akey"])
    results = {s: SceneResult.load(os.path.join(d, f"{s}.npz")) for s in ctx["scenes"]
               if os.path.exists(os.path.join(d, f"{s}.npz"))}
    if not results:
        raise SweepIncomplete(f"variant {vname}: monte_carlo needs the stored scene results in {d}")
    per = {}
    for s in mc_samples(mc, cfg):
        c = copy.deepcopy(cfg)
        for path, val in s.items():
            _set_path(c, path, val)
        pr = Project(cfg=c, base_dir=ctx["proj_dir"])
        F = fields_from_system(pr.system(results))
        p, x = variant_response(pr, F, mode, v.get("offline") or {})
        for r in _rows(F["freqs"].astype(float), p, x, zones, bands, smoothing, vname):
            per.setdefault((r["zone"], r["band"]), []).append(r)
    out = []
    for q in (5, 50, 95):
        for (z, b), rs in per.items():
            row = {"variant": f"{vname}[p{q}]", "zone": z, "band": b, "n_freqs": rs[0]["n_freqs"]}
            for k in MC_METRICS:
                row[k] = float(np.percentile([r[k] for r in rs], q))
            out.append(row)
    return out


# --------------------------------------------------------------------------- sharding (GitHub Actions, Slurm)
def shards(ctx, max_jobs=20):
    """Split the needed solves into independent shards: (solve group, whole frequency bands).

    Shards are unions of whole bands of the full run, so each band keeps the mesh it would get in an
    unsharded run (h from the band's top frequency) and the merged results match a single-machine run
    (exactly for scenes starting at the sweep's first frequency). Cheapest neighbours are merged until
    there are at most max_jobs shards.
    """
    from .scene import _bands
    out = []
    for g in ctx["groups"].values():
        for fb in _bands(ctx["freqs"], ctx["band_octaves"]):
            h = min(ctx["h_max"], ctx["c"] / (fb.max() * ctx["epw"]))
            out.append({"group": g["key"], "freqs": [float(x) for x in fb],
                        "cost": len(fb) * (ctx["h_max"] / h) ** 4 + 30})
    while len(out) > max_jobs:
        pairs = [(out[i]["cost"] + out[i + 1]["cost"], i) for i in range(len(out) - 1)
                 if out[i]["group"] == out[i + 1]["group"]]
        if not pairs:
            break
        _, i = min(pairs)
        b = out.pop(i + 1)
        out[i]["freqs"] += b["freqs"]
        out[i]["cost"] += b["cost"]
    for i, s in enumerate(out):
        s["shard"] = i
    return out


def solve_shard(ctx, i, max_jobs=20):
    """Solve one shard; writes parts/<field key>__<i>.npz for every field of its group."""
    s = shards(ctx, max_jobs)[i]
    g = ctx["groups"][s["group"]]
    parts = os.path.join(ctx["out_dir"], "parts")
    os.makedirs(parts, exist_ok=True)
    fq = np.asarray(s["freqs"])
    fields = [(cfg, pd, sc, fq, pts, os.path.join(parts, os.path.basename(fpath)[:-4] + f"__{i}.npz"))
              for cfg, pd, sc, _, pts, fpath in g["fields"]]
    t0 = time.time()
    _solve_group({"key": g["key"], "fields": fields, "variants": g["variants"],
                  "acoustic_out": os.path.join(parts, f"A{g['key']}__{{scene}}__{i}.npz") if g.get("acoustic") else None})
    return time.time() - t0


def merge_shards(ctx):
    """Join shard parts along frequency into fields/<key>.npz (channels and bodies matched by name)."""
    from .scene import _atomic_savez
    parts = os.path.join(ctx["out_dir"], "parts")
    by_key = {}
    acoustic = {}
    for fn in sorted(os.listdir(parts)) if os.path.isdir(parts) else []:
        if fn.endswith(".npz") and fn.startswith("A") and fn.count("__") == 2:
            akey, scene, _ = fn[1:].split("__")
            acoustic.setdefault((akey, scene), []).append(os.path.join(parts, fn))
        elif fn.endswith(".npz") and "__" in fn:
            by_key.setdefault(fn.split("__")[0], []).append(load_fields(os.path.join(parts, fn)))
    from .jobs import merge as merge_scene
    for (akey, scene), paths in acoustic.items():
        d = os.path.join(ctx["out_dir"], "acoustic", akey)
        os.makedirs(d, exist_ok=True)
        merge_scene(paths).save(os.path.join(d, f"{scene}.npz"), compact=True)
    for akey in {a for a, _ in acoustic}:
        open(os.path.join(ctx["out_dir"], "acoustic", akey, "done"), "w").close()
    for key, P in by_key.items():
        P.sort(key=lambda F: float(F["freqs"][0]))
        f = np.concatenate([F["freqs"] for F in P])
        chans = sorted({str(c) for F in P for c in F["channels"]})
        bodies = sorted({(str(c), str(b)) for F in P for c, b in zip(F["body_channel"], F["body_name"])})
        npts = P[0]["points"].shape[0]
        p = np.zeros((len(chans), len(f), npts), np.complex64)
        pim = np.zeros_like(p)
        cen = np.zeros((len(chans), 3))
        Z = np.full((len(chans), len(f)), np.inf, np.complex64)
        cov = np.zeros((len(chans), len(f)), bool)
        x = np.zeros((len(bodies), len(f)), np.complex64)
        o = 0
        for F in P:
            n = len(F["freqs"])
            for j, c in enumerate(map(str, F["channels"])):
                k = chans.index(c)
                p[k, o:o + n], Z[k, o:o + n], cov[k, o:o + n] = F["p"][j], F["Z"][j], F["covered"][j]
                if "p_img" in F:
                    pim[k, o:o + n] = F["p_img"][j]
                    cen[k] = F["center"][j]
            for j, cb in enumerate(zip(map(str, F["body_channel"]), map(str, F["body_name"]))):
                x[bodies.index(cb), o:o + n] = F["x"][j]
            o += n
        _atomic_savez(os.path.join(ctx["out_dir"], "fields", f"{key}.npz"), freqs=f, points=P[0]["points"],
                      channels=np.array(chans), p=p, p_img=pim, center=cen, Z=Z, covered=cov,
                      body_channel=np.array([c for c, _ in bodies]),
                      body_name=np.array([b for _, b in bodies]), x=x)
    return sorted(by_key)


def _write_csv(path, rows, keys):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def format_reference(rows, label):
    lines = [f"== floor vs reference {label} (aligned by the median offset; sixth-octave bands)",
             f"{'variant':28s} {'band':>9s} {'dMed':>6s} {'spot rms':>8s} {'model p10..p90':>15s} "
             f"{'ref p10..p90':>13s} {'>6dB m/r':>11s}"]
    for r in rows:
        lines.append(f"{r['variant']:28.28s} {r['band']:>9s} {r['d_median_db']:+6.1f} {r['spot_rms_db']:8.2f} "
                     f"{r['model_p10']:+6.1f}..{r['model_p90']:+5.1f}  {r['ref_p10']:+6.1f}..{r['ref_p90']:+5.1f} "
                     f"{r['model_holes6_pct']:5.1f}/{r['ref_holes6_pct']:4.1f}%")
    return "\n".join(lines)


def load_variant(path):
    """Saved variant: freqs, points, total pressure p (nf, npts) and {zone: indices}."""
    z = np.load(path, allow_pickle=False)
    idx = np.split(z["zone_index"], np.cumsum(z["zone_len"])[:-1])
    return {"freqs": z["freqs"], "points": z["points"], "p": z["p"].astype(complex),
            "zones": dict(zip(map(str, z["zones"]), idx))}


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_=.," else "_" for ch in s)
