"""Floor metrics per listener zone and band, matching the design project's floor model.

Definitions (per zone, per band):
  median_db   median over spots of the band level (power average over the band), dB SPL
  p10 / p90   tonal spread: each spot is first normalised to its own overall level (power average over
              all bands analysed); the per-frequency deviation from the zone median, smoothed to 1/n
              octave, is pooled over the band's frequencies and spots
  rms         rms of that pooled tonal deviation
  holes6/3    share of spots where, inside the band, the spot's tonal deviation from the zone median
              (1/n-octave smoothed) dips more than 6 / 3 dB below that spot's own mean deviation over
              the band. Response slopes every spot shares (high-pass, crossover) cancel; only local
              dips count
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
        below = dev[m].mean(0)[None] - dev[m]                   # dip below the spot's own mean tonal deviation       # how far each smoothed frequency sits below the spot's band average
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


# ----------------------------------------------------------------------------- per frequency
def per_frequency(f, p, zones, smoothing=6):
    """Rows [{zone, freq, median_db, p10, p90}]: zone median level (dB SPL) and the tonal spread
    (1/n-octave smoothed, each spot normalised to its own overall level) at every analysis frequency."""
    P2 = np.abs(p) ** 2 + 1e-30
    rows = []
    for z, idx in zones.items():
        Ls = 10 * np.log10(smooth(f, P2[:, idx], smoothing))
        T = Ls - 10 * np.log10(P2[:, idx].mean(0))[None]
        dev = T - np.median(T, axis=1, keepdims=True)
        med = 10 * np.log10(np.median(P2[:, idx], axis=1) / P_REF2)
        p10, p90 = np.percentile(dev, 10, axis=1), np.percentile(dev, 90, axis=1)
        rows += [{"zone": z, "freq": float(fi), "median_db": float(m), "p10": float(a), "p90": float(b)}
                 for fi, m, a, b in zip(f, med, p10, p90)]
    return rows


# ----------------------------------------------------------------------------- band-level data
def octave_band_levels(f, p, centres, n=6):
    """Power-average level (dB SPL) of each spot in 1/n-octave bands at `centres`: (nbands, nspots).
    Bands without an analysis frequency are NaN."""
    P2 = np.abs(p) ** 2 + 1e-30
    out = np.full((len(centres), p.shape[1]), np.nan)
    for i, c in enumerate(centres):
        m = (f >= c * 2 ** (-0.5 / n)) & (f < c * 2 ** (0.5 / n))
        if m.any():
            out[i] = 10 * np.log10(P2[m].mean(0) / P_REF2)
    return out


def band_metrics_from_levels(centres, L, bands):
    """Same metrics as band_metrics, from fractional-octave band levels L (nbands, nspots) in dB
    (e.g. the floor model's sixth-octave table). Holes use the band data's own resolution."""
    ok = ~np.isnan(L).any(1)
    c, L = np.asarray(centres)[ok], L[ok]
    in_any = np.zeros(len(c), bool)
    for lo, hi in bands.values():
        in_any |= (c >= lo) & (c <= hi)
    lin = 10 ** (L / 10)
    spot = 10 * np.log10(lin[in_any].mean(0))
    T = L - spot[None]
    dev = T - np.median(T, axis=1, keepdims=True)
    out = {}
    for name, (lo, hi) in bands.items():
        m = (c >= lo) & (c <= hi)
        if not m.any():
            continue
        band_lvl = 10 * np.log10(lin[m].mean(0))
        d = dev[m].ravel()
        below = dev[m].mean(0)[None] - dev[m]
        out[name] = {"median_db": float(np.median(band_lvl)), "p10": float(np.percentile(d, 10)),
                     "p90": float(np.percentile(d, 90)), "rms": float(np.sqrt(np.mean(d ** 2))),
                     "holes6_pct": float(100 * (below > 6).any(0).mean()),
                     "holes3_pct": float(100 * (below > 3).any(0).mean()), "n_freqs": int(m.sum())}
    return out


def load_reference_csv(path):
    """Floor-model table: '#' comments, header x_m,y_m,z_m,<band centres...>, one row per spot.
    Returns (points (n,3), centres (nb,), levels (nb, n) in dB re arbitrary)."""
    import csv
    with open(path) as fh:
        rows = [r for r in csv.reader(line for line in fh if not line.startswith("#")) if r]
    head = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    return data[:, :3], np.array([float(h) for h in head[3:]]), data[:, 3:].T


def match_points(points, ref_points, tol=1e-3):
    """Index into `points` for every reference point (raises if any is unmatched)."""
    d = np.linalg.norm(points[None, :, :] - ref_points[:, None, :], axis=-1)
    idx = d.argmin(1)
    bad = d[np.arange(len(ref_points)), idx] > tol
    if bad.any():
        raise ValueError(f"{int(bad.sum())} reference spots have no model point within {tol} m "
                         f"(first: {ref_points[bad][0]})")
    return idx


def compare_to_reference(f, p, points, ref_csv, bands, n=6):
    """Model vs reference table, per band, on the reference's own sixth-octave resolution.

    The reference level is arbitrary, so model and reference are aligned by their median
    difference over all spots and bands first ('offset_db'). Rows: band, model/ref metrics,
    d_median_db (after alignment) and spot_rms_db (rms over spots of the tonal difference).
    """
    rp, centres, Lref = load_reference_csv(ref_csv)
    idx = match_points(points, rp)
    Lm = octave_band_levels(f, p[:, idx], centres, n)
    ok = ~np.isnan(Lm).any(1)
    offset = float(np.median(Lm[ok] - Lref[ok]))
    Lr = Lref + offset
    mm = band_metrics_from_levels(centres, np.where(ok[:, None], Lm, np.nan), bands)
    rr = band_metrics_from_levels(centres[ok], Lr[ok], bands)
    sm = Lm[ok] - 10 * np.log10((10 ** (Lm[ok] / 10)).mean(0))[None]
    sr = Lr[ok] - 10 * np.log10((10 ** (Lr[ok] / 10)).mean(0))[None]
    c = centres[ok]
    rows = []
    for b in mm:
        if b not in rr:
            continue
        m = (c >= bands[b][0]) & (c <= bands[b][1])
        rows.append({"band": b, **{f"model_{k}": v for k, v in mm[b].items() if k != "n_freqs"},
                     **{f"ref_{k}": v for k, v in rr[b].items() if k != "n_freqs"},
                     "d_median_db": mm[b]["median_db"] - rr[b]["median_db"],
                     "spot_rms_db": float(np.sqrt(np.mean((sm[m] - sr[m]) ** 2)))})
    return {"offset_db": offset, "n_spots": len(idx), "rows": rows}


def diff_rows(rows_a, rows_b, keys=("median_db", "p10", "p90", "rms", "holes6_pct", "holes3_pct")):
    """Difference table b - a for rows matched on (zone, band)."""
    A = {(r["zone"], r["band"]): r for r in rows_a}
    out = []
    for r in rows_b:
        a = A.get((r["zone"], r["band"]))
        if a is None:
            continue
        out.append({"zone": r["zone"], "band": r["band"],
                    **{f"d_{k}": float(r[k]) - float(a[k]) for k in keys if k in r and k in a}})
    return out
