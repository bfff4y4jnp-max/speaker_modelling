"""DSP transfer functions and CamillaDSP config import (engineering convention).

All filters are evaluated exactly in the z-domain at the analysis
frequencies, so the modelled crossover is the one that will run.
"""
from __future__ import annotations

import numpy as np


def biquad(f, fs, b, a):
    z1 = np.exp(-2j * np.pi * np.asarray(f, float) / fs)
    return (b[0] + b[1] * z1 + b[2] * z1 ** 2) / (a[0] + a[1] * z1 + a[2] * z1 ** 2)


def rbj(kind, f0, fs, q=0.7071, gain_db=0.0):
    """RBJ cookbook coefficients (b, a) as used by CamillaDSP."""
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    al = sw / (2 * q)
    A = 10 ** (gain_db / 40)
    if kind == "Lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "Highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "Bandpass":
        b = [al, 0, -al]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "Notch":
        b = [1, -2 * cw, 1]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "Allpass":
        b = [1 - al, -2 * cw, 1 + al]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "Peaking":
        b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind in ("Lowshelf", "Highshelf"):
        s = 2 * np.sqrt(A) * al
        if kind == "Lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + s), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s)]
            a = [(A + 1) + (A - 1) * cw + s, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + s), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s)]
            a = [(A + 1) - (A - 1) * cw + s, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s]
    elif kind == "LowpassFO":
        k = np.tan(w0 / 2); b = [k, k, 0]; a = [1 + k, k - 1, 0]
    elif kind == "HighpassFO":
        k = np.tan(w0 / 2); b = [1, -1, 0]; a = [1 + k, k - 1, 0]
    else:
        raise ValueError(f"unsupported biquad type {kind}")
    b, a = np.array(b, float), np.array(a, float)
    return b / a[0], a / a[0]


def butterworth_qs(order):
    """Q values of the second-order sections (odd orders add a first-order section)."""
    return [1 / (2 * np.sin((2 * k + 1) * np.pi / (2 * order))) for k in range(order // 2)]


def crossover(f, fs, kind, freq, order):
    """ButterworthLowpass/Highpass, LinkwitzRileyLowpass/Highpass of any CamillaDSP order."""
    lp = "Lowpass" in kind
    if kind.startswith("LinkwitzRiley"):
        h = crossover(f, fs, "Butterworth" + ("Lowpass" if lp else "Highpass"), freq, order // 2)
        return h * h
    H = np.ones(len(np.atleast_1d(f)), complex)
    for q in butterworth_qs(order):
        H = H * biquad(f, fs, *rbj("Lowpass" if lp else "Highpass", freq, fs, q))
    if order % 2:
        H = H * biquad(f, fs, *rbj("LowpassFO" if lp else "HighpassFO", freq, fs))
    return H


def delay(f, seconds):
    return np.exp(-2j * np.pi * np.asarray(f, float) * seconds)


def fir(f, fs, taps):
    taps = np.asarray(taps, float)
    z = np.exp(-2j * np.pi * np.outer(np.asarray(f, float), np.arange(len(taps))) / fs)
    return z @ taps


def filter_response(f, fs, spec, base_dir="."):
    """Response of one CamillaDSP filter definition (dict with type/parameters)."""
    t, p = spec["type"], spec.get("parameters", {})
    f = np.asarray(f, float)
    if t == "Biquad":
        st = p["type"]
        if st == "Free":
            return biquad(f, fs, [p["b0"], p["b1"], p["b2"]], [1.0, p["a1"], p["a2"]])
        q = p.get("q")
        if q is None and "bandwidth" in p:
            bw = p["bandwidth"]; w0 = 2 * np.pi * p["freq"] / fs
            q = 1 / (2 * np.sinh(np.log(2) / 2 * bw * w0 / np.sin(w0)))
        if q is None and "slope" in p:
            A = 10 ** (p["gain"] / 40)
            q = 1 / np.sqrt((A + 1 / A) * (1 / (p["slope"] / 12.0) - 1) + 2)
        return biquad(f, fs, *rbj(st, p["freq"], fs, q if q is not None else 0.7071, p.get("gain", 0.0)))
    if t == "BiquadCombo":
        return crossover(f, fs, p["type"], p["freq"], p["order"])
    if t == "Delay":
        unit = p.get("unit", "ms")
        d = p["delay"]
        sec = d / 1000 if unit == "ms" else d / fs if unit == "samples" else d / 1000 / 343.0
        return delay(f, sec)
    if t == "Gain":
        g = 10 ** (p.get("gain", 0.0) / 20) if p.get("scale", "dB") == "dB" else p.get("gain", 1.0)
        g = -g if p.get("inverted") else g
        return np.full(len(f), 0.0 if p.get("mute") else g, complex)
    if t == "Volume" or t == "Loudness" or t == "Dither" or t == "Limiter" or t == "Compressor":
        return np.ones(len(f), complex)          # level/dynamics: small-signal unity
    if t == "Conv":
        ct = p["type"]
        if ct == "Values":
            return fir(f, fs, p["values"])
        import os
        path = p["filename"] if os.path.isabs(p["filename"]) else os.path.join(base_dir, p["filename"])
        if ct == "Raw":
            fmt = p.get("format", "TEXT")
            if fmt == "TEXT":
                taps = np.loadtxt(path)
            else:
                dt = {"FLOAT32LE": "<f4", "FLOAT64LE": "<f8", "S16LE": "<i2", "S32LE": "<i4"}[fmt]
                taps = np.fromfile(path, dtype=dt).astype(float)
                if fmt.startswith("S"):
                    taps /= 2.0 ** (8 * np.dtype(dt).itemsize - 1)
            return fir(f, fs, taps)
        if ct == "Wav":
            from scipy.io import wavfile
            _, data = wavfile.read(path)
            data = data[:, p.get("channel", 0)] if data.ndim > 1 else data
            if data.dtype.kind == "i":
                data = data / 2.0 ** (8 * data.dtype.itemsize - 1)
            return fir(f, fs, data)
    if t == "DiffEq":
        a, b = p.get("a", [1.0]), p.get("b", [1.0])
        z1 = np.exp(-2j * np.pi * f / fs)
        num = sum(bk * z1 ** k for k, bk in enumerate(b))
        den = sum(ak * z1 ** k for k, ak in enumerate(a))
        return num / den
    raise ValueError(f"unsupported CamillaDSP filter type {t}")


def camilladsp_transfer(config, f, base_dir=None):
    """Transfer matrix H[out, in](f) of a CamillaDSP v1/v2 config (path or dict).

    Returns (H (nf, n_out, n_in), samplerate).
    """
    import os
    import yaml
    if isinstance(config, str):
        base_dir = base_dir or os.path.dirname(os.path.abspath(config))
        with open(config) as fh:
            config = yaml.safe_load(fh)
    base_dir = base_dir or "."
    fs = config["devices"]["samplerate"]
    n = config["devices"]["capture"]["channels"]
    f = np.asarray(f, float)
    H = np.broadcast_to(np.eye(n), (len(f), n, n)).astype(complex)
    filters = config.get("filters", {}) or {}
    mixers = config.get("mixers", {}) or {}
    for step in config.get("pipeline", []):
        if step.get("bypassed"):
            continue
        if step["type"] == "Mixer":
            m = mixers[step["name"]]
            M = np.zeros((m["channels"]["out"], m["channels"]["in"]))
            for mp in m["mapping"]:
                if mp.get("mute"):
                    continue
                for s in mp["sources"]:
                    if s.get("mute"):
                        continue
                    g = 10 ** (s.get("gain", 0.0) / 20) if s.get("scale", "dB") == "dB" else s.get("gain", 1.0)
                    M[mp["dest"], s["channel"]] += -g if s.get("inverted") else g
            H = np.einsum("oi,fij->foj", M, H)
        elif step["type"] == "Filter":
            chans = step.get("channels", [step.get("channel")] if "channel" in step else range(H.shape[1]))
            for ch in chans:
                for name in step["names"]:
                    H[:, ch, :] *= filter_response(f, fs, filters[name], base_dir)[:, None]
    return H, fs
