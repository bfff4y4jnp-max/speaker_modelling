"""System level: scenes + drivers + DSP + amps + ground + listener metrics.

Everything here is cheap linear algebra on stored scene results, so DSP,
delays, wiring and presets can be iterated instantly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .air import Air, ground_reflection
from .drivers import Wiring, solve_network

P_REF = 20e-6


@dataclass
class SceneEntry:
    name: str
    result: object
    wiring: Wiring
    ground_sigma: float = np.inf      # impedance ground for scenes solved without a BEM ground (needs T_img)


@dataclass
class SystemModel:
    air: Air = field(default_factory=Air)
    scenes: list = field(default_factory=list)
    absorption: bool = True

    def add_scene(self, name, result, wiring, ground_sigma=np.inf):
        self.scenes.append(SceneEntry(name, result, wiring, ground_sigma))
        return self

    @property
    def freqs(self):
        """Union of all scene grids (scenes from one project share a log grid)."""
        return np.unique(np.concatenate([sc.result.freqs for sc in self.scenes]).round(6))

    @property
    def points(self):
        return self.scenes[0].result.points

    def channel_responses(self):
        """{channel: {'p': (nf,npts) Pa per volt, 'x'/'i': {body: per volt}, 'Z': input impedance}}.

        Scenes contribute only over their solved range (zero elsewhere);
        'covered' marks where every scene feeding the channel was solved.
        """
        out = {}
        f = self.freqs
        nf = len(f)
        for sc in self.scenes:
            r = sc.result
            idx = np.searchsorted(f, r.freqs.round(6))
            net = solve_network(r, sc.wiring, self.air)
            for ch, d in net.items():
                p = d["p"].copy()
                if r.T_img is not None and not r.ground:
                    p = p + self._ground_factor(r, sc.ground_sigma, r.freqs) * d["p_img"]
                if self.absorption:
                    dist = np.linalg.norm(r.points - r.center, axis=1)
                    p = p * np.exp(-np.outer(self.air.absorption(r.freqs), dist))
                o = out.setdefault(ch, {"p": np.zeros((nf, p.shape[1]), complex), "x": {}, "i": {},
                                        "Y": np.zeros(nf, complex), "covered": np.ones(nf, bool)})
                o["p"][idx] += p
                cov = np.zeros(nf, bool); cov[idx] = True
                o["covered"] &= cov
                for j, b in enumerate(r.body_names):
                    if d["i"][:, j].any():
                        x = np.zeros(nf, complex); x[idx] = d["x"][:, j]; o["x"][b] = x
                        i = np.zeros(nf, complex); i[idx] = d["i"][:, j]; o["i"][b] = i
                o["Y"][idx] += 1 / d["Z"]
        for o in out.values():   # parallel combination across scenes sharing a channel
            o["Z"] = np.where(o["Y"] != 0, 1 / np.where(o["Y"] != 0, o["Y"], 1), np.inf)
        return out

    def _ground_factor(self, r, sigma, f):
        s = r.center
        img = s * [1, 1, -1]
        d = r.points - img
        r2 = np.linalg.norm(d, axis=1)
        cos = (r.points[:, 2] + s[2]) / r2
        return ground_reflection(f[:, None], r2[None], cos[None], sigma, self.air)

    def response(self, channel_tf, ch_resp=None):
        """Total pressure (nf, npts) for program input.

        channel_tf: {channel: (nf,) volts at the amp output per unit input}
        (DSP x amp gain), e.g. from `channel_transfer`.
        """
        ch_resp = ch_resp or self.channel_responses()
        p = 0
        for ch, d in ch_resp.items():
            if ch in channel_tf:
                p = p + d["p"] * channel_tf[ch][:, None]
        return p

    def max_input(self, channel_tf, v_max, x_max, ch_resp=None):
        """Largest input amplitude (per frequency) before any channel voltage or excursion limit.

        v_max: {channel: rms volts}; x_max: {body name: peak metres} (phasors are rms).
        Returns (limit (nf,), name of limiting element per frequency).
        """
        ch_resp = ch_resp or self.channel_responses()
        f = self.freqs
        lim = np.full(len(f), np.inf)
        who = np.array([""] * len(f), dtype=object)
        x_tot = {}
        for ch, d in ch_resp.items():
            if ch not in channel_tf:
                continue
            g = np.abs(channel_tf[ch])
            if ch in v_max:
                l = v_max[ch] / np.maximum(g, 1e-30)
                upd = l < lim; lim[upd] = l[upd]; who[upd] = f"amp:{ch}"
            for b, x in d["x"].items():
                x_tot[b] = x_tot.get(b, 0) + x * channel_tf[ch]
        for b, x in x_tot.items():
            if b in x_max:
                l = x_max[b] / np.maximum(np.sqrt(2) * np.abs(x), 1e-30)
                upd = l < lim; lim[upd] = l[upd]; who[upd] = f"xmax:{b}"
        return lim, who


def spl(p):
    """dB SPL of rms phasors (all spkmod phasors are rms)."""
    return 20 * np.log10(np.maximum(np.abs(p), 1e-30) / P_REF)


def frac_octave(f, P2, n=6, fmin=None, fmax=None):
    """Power average of |p|^2 (nf, ...) in 1/n-octave bands. Returns (centres, band power)."""
    f = np.asarray(f)
    fmin = fmin or f.min()
    fmax = fmax or f.max()
    k = np.arange(np.ceil(n * np.log2(fmin / 1000)), np.floor(n * np.log2(fmax / 1000)) + 1)
    fc = 1000 * 2 ** (k / n)
    out = []
    keep = []
    for c in fc:
        m = (f >= c * 2 ** (-0.5 / n)) & (f < c * 2 ** (0.5 / n))
        if m.any():
            out.append(P2[m].mean(0))
            keep.append(c)
    return np.array(keep), np.array(out)


DEFAULT_BANDS = {"40-90": (40, 90), "90-180": (90, 180), "180-450": (180, 450), "450-1k": (450, 1000),
                 "1k-2.5k": (1000, 2500), "2.5k-6k": (2500, 6000), "6k-12k": (6000, 12000)}


def floor_stats(f, p, bands=DEFAULT_BANDS, norm_range=(50, 8000), dip_db=6.0, n=6):
    """Tonal-consistency statistics over listener points.

    Each spot is normalised to its own overall level over `norm_range`.
    spread: p10/p90 of the spot deviation from the floor median (1/n octave).
    dips: share of spots where some analysis frequency inside the band is
    more than `dip_db` below that spot's own band average.
    """
    P2 = np.abs(p) ** 2
    m = (f >= norm_range[0]) & (f <= norm_range[1])
    spot = 10 * np.log10(P2[m].mean(0))
    fc, B = frac_octave(f, P2, n)
    LdB = 10 * np.log10(B) - spot[None]
    med = np.median(LdB, axis=1)
    dev = LdB - med[:, None]
    out = {"centres": fc, "median": med + np.median(spot), "rows": {}}
    for name, (lo, hi) in bands.items():
        sel = (fc >= lo) & (fc <= hi)
        fm = (f >= lo) & (f <= hi)
        if not sel.any() or not fm.any():
            continue
        d = dev[sel].ravel()
        band_avg = 10 * np.log10(P2[fm].mean(0))
        dips = (10 * np.log10(P2[fm]) < band_avg[None] - dip_db).any(0)
        out["rows"][name] = {"p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90)),
                             "dips_pct": float(100 * dips.mean())}
    return out


def listener_grid(x_range, y_range, spacing, heights=(1.4, 1.6, 1.8)):
    xs = np.arange(x_range[0], x_range[1] + 1e-9, spacing)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, spacing)
    X, Y, Z = np.meshgrid(xs, ys, heights, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def log_freqs(fmin, fmax, per_octave=24):
    n = int(np.ceil(np.log2(fmax / fmin) * per_octave)) + 1
    return fmin * 2 ** (np.arange(n) / per_octave)
