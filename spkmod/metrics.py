"""Floor metrics per listener zone and band, matching the design project's floor model.

Definitions (per zone, per band):
  median_db   median over spots of the band level (power average over the band), dB SPL
  p10 / p90   tonal spread: each spot is first normalised to its own overall level (power average over
              all bands analysed); the per-frequency deviation from the zone median, smoothed to 1/n
              octave, is pooled over the band's frequencies and spots
  rms         rms of that pooled tonal deviation
  holes6/3    share of spots where some (1/n-octave smoothed) frequency in the band is more than
              6 / 3 dB below that spot's own average over the band
"""
from __future__ import annotations

import numpy as np

P_REF2 = 4e-10                     # (20 uPa)^2
FLOOR_BANDS = {"40-90": (40, 90), "90-180": (90, 180), "180-450": (180, 450), "450-1k": (450, 1000),
               "1k-2.5k": (1000, 2500), "2.5k-6k": (2500, 6000), "6k-12k": (6000, 12000)}
SUB_BANDS = {"25-30": (25, 30), "30-36": (30, 36), "36-45": (36, 45), "45-60": (45, 60), "60-80": (60, 80),
             "80-100": (80, 100)}


def parse_bands(spec, f):
    """Band set from a spec: None (floor-model + sub bands), 'sixth'/'third'/'octave', a list of
    [lo, hi] or a {name: [lo, hi]} dict. Bands with no analysis frequency are dropped."""
    if spec is None:
        bands = {**SUB_BANDS, **FLOOR_BANDS}
    elif isinstance(spec, str):
        n = {"sixth": 6, "third": 3, "octave": 1}[spec]
        k = np.arange(np.ceil(n * np.log2(f.min() / 1000)), np.floor(n * np.log2(f.max() / 1000)) + 1)
        bands = {f"{c:.4g}": (c * 2 ** (-0.5 / n), c * 2 ** (0.5 / n)) for c in 1000 * 2 ** (k / n)}
    elif isinstance(spec, dict):
        bands = {k: tuple(v) for k, v in spec.items()}
    else:
        bands = {f"{lo:g}-{hi:g}": (lo, hi) for lo, hi in spec}
    return {k: v for k, v in bands.items() if ((f >= v[0]) & (f <= v[1])).any()}


def smooth(f, P2, n):
    """1/n-octave power smoothing along axis 0 (P2 = |p|^2)."""
    if not n:
        return P2
    out = np.empty_like(P2)
    lf = np.log2(f)
    for i, x in enumerate(lf):
        m = np.abs(lf - x) <= 0.5 / n
        out[i] = P2[m].mean(0)
    return out


def band_metrics(f, p, bands, smoothing=6):
    """{band: metrics} for one set of spots. p: (nf, nspots) complex pressure (rms phasors)."""
    f = np.asarray(f)
    P2 = np.abs(p) ** 2 + 1e-30
    Ls = 10 * np.log10(smooth(f, P2, smoothing))
    in_any = np.zeros(len(f), bool)
    for lo, hi in bands.values():
        in_any |= (f >= lo) & (f <= hi)
    spot = 10 * np.log10(P2[in_any].mean(0))                   # each spot's own overall level
    T = Ls - spot[None]
    dev = T - np.median(T, axis=1, keepdims=True)
    out = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        if not m.any():
            continue
        band_lvl = 10 * np.log10(P2[m].mean(0) / P_REF2)       # dB SPL
        d = dev[m].ravel()
        below = (band_lvl[None] + 10 * np.log10(P_REF2)) - Ls[m]       # how far each smoothed frequency sits below the spot's band average
        out[name] = {"median_db": float(np.median(band_lvl)), "p10": float(np.percentile(d, 10)),
                     "p90": float(np.percentile(d, 90)), "rms": float(np.sqrt(np.mean(d ** 2))),
                     "holes6_pct": float(100 * (below > 6).any(0).mean()),
                     "holes3_pct": float(100 * (below > 3).any(0).mean()), "n_freqs": int(m.sum())}
    return out


def zone_table(f, p, zones, bands, smoothing=6):
    """Rows [{zone, band, ...metrics}] for every zone (name -> spot indices)."""
    rows = []
    for z, idx in zones.items():
        for b, m in band_metrics(f, p[:, idx], bands, smoothing).items():
            rows.append({"zone": z, "band": b, **m})
    return rows


def zone_points(spec, base_dir="."):
    """Points of one zone: {box: [x0,x1,y0,y1], heights: [...], spacing: s} | {file: csv} | {points: [...]}."""
    import os
    if "box" in spec:
        x0, x1, y0, y1 = spec["box"]
        s = spec.get("spacing", 0.5)
        xs = np.arange(x0, x1 + 1e-9, s)
        ys = np.arange(y0, y1 + 1e-9, s)
        X, Y, Z = np.meshgrid(xs, ys, spec.get("heights", [1.6]), indexing="ij")
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    if "file" in spec:
        path = spec["file"] if os.path.isabs(spec["file"]) else os.path.join(base_dir, spec["file"])
        return np.atleast_2d(np.loadtxt(path, delimiter=","))
    return np.atleast_2d(np.asarray(spec["points"], float))


def format_table(rows, keys=("median_db", "p10", "p90", "holes6_pct", "holes3_pct"), group=("zone",)):
    """Plain-text table, one block per zone, one line per (variant, band)."""
    lines = []
    for z in dict.fromkeys(r["zone"] for r in rows):
        lines.append(f"== zone {z}")
        lines.append(f"{'variant':28s} {'band':>9s} {'median':>7s} {'p10..p90':>13s} {'rms':>5s} {'>6dB':>6s} {'>3dB':>6s}"
                     + (f" {'dMed':>6s}" if any('d_median_db' in r for r in rows) else "")
                     + (f" {'xpk mm':>7s}" if any('x_peak_mm' in r for r in rows) else ""))
        for r in rows:
            if r["zone"] != z:
                continue
            line = (f"{r.get('variant', ''):28.28s} {r['band']:>9s} {r['median_db']:7.1f} "
                    f"{r['p10']:+6.1f}..{r['p90']:+5.1f} {r['rms']:5.2f} {r['holes6_pct']:5.1f}% {r['holes3_pct']:5.1f}%")
            if "d_median_db" in r:
                line += f" {r['d_median_db']:+6.1f}"
            if "x_peak_mm" in r:
                line += f" {r['x_peak_mm']:7.2f}"
            lines.append(line)
    return "\n".join(lines)
