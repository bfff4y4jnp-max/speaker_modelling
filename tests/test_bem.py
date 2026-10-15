"""Analytic validation of the exterior BEM."""
import numpy as np
import pytest

import bempp_cl.api as _b
_b.set_logging_level("warning")

from spkmod.bem import ExteriorBEM


def sphere(n_ref=3, r=1.0, hemi=False):
    import bempp_cl.api as bempp
    g = bempp.shapes.regular_sphere(n_ref)
    v = g.vertices.T * r
    t = g.elements.T.astype(np.int64)
    if hemi:  # keep upper half, add flat disc bottom on z=0 (removed by ground folding)
        v[:, 2][np.abs(v[:, 2]) < 1e-9] = 0.0
        keep = (v[t][:, :, 2] >= -1e-12).all(1)
        t = t[keep]
        used = np.unique(t)
        # close with a fan at z=0
        rim = used[np.abs(v[used, 2]) < 1e-9]
        ang = np.arctan2(v[rim, 1], v[rim, 0])
        rim = rim[np.argsort(ang)]
        c = len(v)
        v = np.vstack([v, [[0, 0, 0]]])
        fan = np.array([[c, rim[(i + 1) % len(rim)], rim[i]] for i in range(len(rim))])
        t = np.vstack([t, fan])
        used = np.unique(t)
        remap = -np.ones(len(v), int)
        remap[used] = np.arange(len(used))
        v, t = v[used], remap[t]
    return v, t


def pulsating(k, r, a=1.0, rho=1.2, c=343.0):
    w = k * c
    A = 1j * w * rho / (np.exp(1j * k * a) * (1j * k - 1 / a) / a)
    return A * np.exp(1j * k * r) / r


@pytest.mark.parametrize("hemi", [False, True])
def test_pulsating_sphere(hemi):
    v, t = sphere(3, hemi=hemi)
    cen = v[t].mean(1)
    if hemi:
        active = np.where(cen[:, 2] > 1e-9)[0]
    else:
        active = np.arange(len(t))
    bem = ExteriorBEM(v, t, active, ground=0.0 if hemi else None)
    k, rho, c = 2.0, 1.2, 343.0
    band = bem.band(1.5, 2.5)
    P = band.solve(k)
    g = 1j * k * c * rho * np.ones(len(active))
    p = P @ g
    pv = bem.to_orig(p)
    pa = pulsating(k, 1.0)
    on = np.linalg.norm(v, axis=1) > 0.99
    on = on & np.isin(np.arange(len(v)), bem.used)
    assert np.abs(pv[on] / pa - 1).max() < 0.02
    x = np.array([[3.0, 0, 0.5], [0, -2.0, 1.0]])
    Dm, Sm = bem.field_matrices(x, k)
    pf = Dm @ p - Sm @ g
    ref = pulsating(k, np.linalg.norm(x, axis=1))
    assert np.abs(pf / ref - 1).max() < 0.02
    # far field: p ~ e^{ikr}/r * pattern ; pattern = A
    Df, Sf = bem.field_matrices(np.array([[0, 0, 1.0]]), k, far=True)
    A = pulsating(k, 1.0) * np.exp(-1j * k)
    assert abs((Df @ p - Sf @ g)[0] / A - 1) < 0.02


def test_opencl_failure_falls_back_to_numba(monkeypatch):
    import spkmod.bem as bem
    monkeypatch.setattr(bem, "_DEVICE", "opencl")
    calls = []

    def make(d):
        calls.append(d)
        if d == "opencl":
            raise RuntimeError("clBuildProgram failed: BUILD_PROGRAM_FAILURE\nerror: unknown target CPU 'generic'")
        return d

    monkeypatch.setattr(bem, "_dense", lambda x: x)
    assert bem._with_fallback(make) == "numba" and calls == ["opencl", "numba"] and bem.device() == "numba"
