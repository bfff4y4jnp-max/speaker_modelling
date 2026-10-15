"""YAML project files: drivers, cabinets (STEP), scenes, wiring, DSP, listeners.

Lengths are metres unless the key carries a unit suffix (_mm, _cm2, _l, ...).
See examples/party_rig/project.yml for a complete annotated example.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import fields

import numpy as np
import yaml

from .air import Air
from .drivers import Driver, PassiveRadiator, Wiring
from .geometry import Cabinet, Piston
from .scene import SolverSettings, solve_scene
from .system import SystemModel, listener_grid, log_freqs

log = logging.getLogger("spkmod")


_INFO_KEYS = ("source", "notes", "note", "comment")


def _driver(name, d, air):
    d = {k: v for k, v in dict(d or {}).items() if k not in _INFO_KEYS}   # documentation-only keys
    missing = [k for k, v in d.items() if v is None]
    if missing or not d:
        raise ValueError(f"driver {name!r}: fill in {missing or 'its parameters'} from the datasheet / measurement")
    if d.pop("passive", False):
        return PassiveRadiator(name, Mmd=d.get("Mmd_g", 0) / 1e3 or d["Mmd"], Cms=d.get("Cms_mm_per_N", 0) / 1e3 or d["Cms"],
                               Rms=d.get("Rms", 0.0))
    sd = d.pop("Sd_cm2", None)
    if sd is not None:
        d["Sd"] = sd / 1e4
    conv = {"Le_mH": ("Le", 1e-3), "L2_mH": ("L2", 1e-3), "Mms_g": ("Mms", 1e-3), "Mmd_g": ("Mmd", 1e-3),
            "Cms_mm_per_N": ("Cms", 1e-3), "Xmax_mm": ("Xmax", 1e-3), "Vas_l": ("Vas", 1e-3)}
    for k, (nk, s) in conv.items():
        if k in d:
            d[nk] = d.pop(k) * s
    if {"Fs", "Qes", "Qms", "Vas"} <= d.keys() and "Bl" not in d:
        return Driver.from_ts(name, d.pop("Fs"), d.pop("Qes"), d.pop("Qms"), d.pop("Vas"), d.pop("Re"), d.pop("Sd"),
                              air=air, **{k: v for k, v in d.items() if k in {f.name for f in fields(Driver)}})
    for k in ("Fs", "Qes", "Qms", "Qts", "Vas"):
        d.pop(k, None)
    return Driver(name=name, **d)


class Project:
    def __init__(self, path=None, cfg=None, base_dir=None):
        """Load project.yml from `path`, or use an in-memory `cfg` dict (paths relative to base_dir)."""
        if cfg is None:
            self.path = os.path.abspath(path)
            with open(path) as fh:
                cfg = yaml.safe_load(fh)
        else:
            self.path = os.path.abspath(path or os.path.join(base_dir or ".", "project.yml"))
        self.dir = os.path.abspath(base_dir) if base_dir else os.path.dirname(self.path)
        self.cfg = cfg
        c = self.cfg
        self.air = Air(**c.get("air", {}))
        s = dict(c.get("solver", {}))
        if s.get("cache_dir"):
            s["cache_dir"] = self._p(s["cache_dir"])
        self.settings = SolverSettings(**s)
        self.drivers = {n: _driver(n, d, self.air) for n, d in (c.get("drivers") or {}).items()}

    def _p(self, p):
        return p if os.path.isabs(p) else os.path.join(self.dir, p)

    # ------------------------------------------------------------------ build
    def freqs(self, scene=None, fmin=None, fmax=None, per_octave=None):
        fc = dict(self.cfg.get("frequencies", {}))
        sc = (self.cfg["scenes"][scene] if scene else {}) or {}
        lo = fmin or sc.get("min_freq") or fc.get("min", 20)
        hi = fmax or sc.get("max_freq") or fc.get("max", 1000)
        f = log_freqs(fc.get("min", 20), fc.get("max", 1000), per_octave or fc.get("per_octave", 24))
        return f[(f >= lo * 0.999) & (f <= hi * 1.001)]

    def listeners(self):
        L = self.cfg.get("listeners")
        if not L:
            return None
        if "grid" in L:
            g = L["grid"]
            return listener_grid(g["x"], g["y"], g["spacing"], g.get("heights", (1.6,)))
        if "file" in L:
            return np.loadtxt(self._p(L["file"]), delimiter=",")
        return np.asarray(L["points"], float)

    def observation_points(self):
        """(points, zones): the listeners as zone 'floor' followed by the project's `zones:`."""
        from .metrics import zone_points
        pts, zones, n = [], {}, 0
        L = self.listeners()
        if L is not None:
            pts.append(L)
            zones["floor"] = np.arange(len(L))
            n = len(L)
        for z, zs in (self.cfg.get("zones") or {}).items():
            P = zone_points(zs, self.dir)
            pts.append(P)
            zones[z] = np.arange(n, n + len(P))
            n += len(P)
        return (np.vstack(pts) if pts else None), zones

    def directions(self, step_deg=5):
        """Horizontal (xy) and vertical (xz) polar directions about +x (front)."""
        a = np.radians(np.arange(-180, 180, step_deg))
        hor = np.column_stack([np.cos(a), np.sin(a), 0 * a])
        ver = np.column_stack([np.cos(a), 0 * a, np.sin(a)])
        return np.vstack([hor, ver])

    def cabinets(self, scene):
        sc = self.cfg["scenes"][scene]
        out = []
        for it in sc["items"]:
            base = copy.deepcopy(self.cfg["cabinets"][it["cabinet"]])
            pistons = [Piston(name=p["name"], center=p["center"], axis=p["axis"],
                              diameter=p.get("diameter") or 2 * np.sqrt(p["sd_cm2"] / 1e4 / np.pi),
                              cutout=p.get("cutout"), thickness=p.get("thickness", 0.008), motor=p.get("motor"))
                       for p in base.get("pistons", [])]
            stuffing = [(s["box"], s["sigma"]) for s in base.get("stuffing", [])]
            out.append(Cabinet(name=it.get("name", it["cabinet"]), step=self._p(base["step"]) if base.get("step") else None,
                               pistons=pistons, stuffing=stuffing, envelope=base.get("envelope"),
                               position=it.get("position", (0, 0, 0)), rotation=tuple(it.get("rotation", (0, 0, 0)))))
        return out

    def solve(self, scene, fmin=None, fmax=None, per_octave=None):
        sc = self.cfg["scenes"][scene]
        ground = sc.get("ground", True)
        settings = self.scene_settings(scene)
        return solve_scene(self.cabinets(scene), self.freqs(scene, fmin, fmax, per_octave), points=self.observation_points()[0],
                           directions=self.directions() if sc.get("polars", True) else None, ground=ground,
                           image_points=not ground, air=self.air, settings=settings,
                           center=sc.get("acoustic_center"))

    def scene_settings(self, scene):
        """Solver settings with the scene's own overrides (scene key `exterior: baffle`)."""
        import dataclasses
        sc = self.cfg["scenes"][scene]
        return dataclasses.replace(self.settings, exterior=sc.get("exterior", self.settings.exterior))

    def wiring(self, scene):
        w = Wiring()
        for body, e in (self.cfg.get("wiring", {}).get(scene) or {}).items():
            w.add(body, self.drivers[e["driver"]], e.get("channel"), e.get("polarity", 1.0))
        return w

    def system(self, results):
        """SystemModel from {scene: SceneResult} (results must share frequencies)."""
        m = SystemModel(air=self.air)
        for name, res in results.items():
            sc = self.cfg["scenes"][name]
            g = sc.get("ground_reflection", sc.get("ground_sigma", np.inf))    # |R|(f) table or flow resistivity
            m.add_scene(name, res, self.wiring(name), g)
        return m

    def channel_tf(self, freqs, preset=None):
        """{channel: volts rms per unit (rms, full-scale) program input}."""
        d = self.cfg.get("dsp") or {}
        amp = d.get("volts_at_0dbfs", 1.0)
        if preset:
            d = {**d, **(d.get("presets", {}).get(preset, {}))}
        if d.get("camilladsp"):
            from .dsp import camilladsp_transfer
            H, _ = camilladsp_transfer(self._p(d["camilladsp"]), freqs)
            wts = np.asarray(d.get("input", [1.0] + [0.0] * (H.shape[2] - 1)), float)
            Hm = H @ wts
            return {ch: amp * Hm[:, i] for ch, i in d["outputs"].items()}
        return {ch: amp * np.ones(len(freqs), complex) for ch in d.get("outputs", {})}
