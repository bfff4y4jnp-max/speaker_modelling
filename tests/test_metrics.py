import numpy as np

from spkmod.metrics import band_metrics, parse_bands, zone_table
from spkmod.system import log_freqs


def test_flat_field_has_no_spread_or_holes():
    f = log_freqs(25, 2000, 24)
    p = np.full((len(f), 30), 2.0 + 0j)
    m = band_metrics(f, p, parse_bands(None, f))
    assert set(m) >= {"25-30", "40-90", "450-1k"}
    for r in m.values():
        assert abs(r["p10"]) < 1e-9 and abs(r["p90"]) < 1e-9 and r["holes6_pct"] == 0
        assert abs(r["median_db"] - 20 * np.log10(2 / 20e-6)) < 1e-6


def test_notch_counts_as_hole_and_level_offset_is_not_tonal():
    f = log_freqs(30, 300, 48)
    p = np.ones((len(f), 10), complex)
    p[:, 5:] *= 4.0                              # level difference only: no tonal spread
    p[np.argmin(abs(f - 70)), 0] = 0.1           # deep narrow notch at spot 0
    m = band_metrics(f, p, {"45-90": (45, 90), "90-180": (90, 180)}, smoothing=0)
    assert m["45-90"]["holes6_pct"] == 10.0 and m["90-180"]["holes6_pct"] == 0
    assert abs(m["90-180"]["p90"]) < 0.05        # only the notch spot's own normalisation moves


def test_zone_table_rows():
    f = log_freqs(40, 100, 12)
    p = np.ones((len(f), 6), complex)
    rows = zone_table(f, p, {"a": np.arange(3), "b": np.arange(3, 6)}, {"40-90": (40, 90)})
    assert [r["zone"] for r in rows] == ["a", "b"]


def test_reference_roundtrip_recovers_offset(tmp_path):
    from spkmod.metrics import compare_to_reference, octave_band_levels, parse_bands
    rng = np.random.default_rng(1)
    f = log_freqs(30, 600, 24)
    pts = np.column_stack([rng.uniform(1, 5, 40), rng.uniform(-2, 2, 40), np.full(40, 1.6)])
    p = (1 + 0.3 * rng.standard_normal((len(f), 40))) * np.exp(1j * rng.uniform(0, 6, (len(f), 40)))
    centres = 1000 * 2 ** (np.arange(-30, -4) / 6)
    L = octave_band_levels(f, p, centres)
    ok = ~np.isnan(L).any(1)
    path = tmp_path / "ref.csv"
    with open(path, "w") as fh:
        fh.write("# test reference\nx_m,y_m,z_m," + ",".join(f"{c:.1f}" for c in centres[ok]) + "\n")
        for k in range(40)[::-1]:                                  # different spot order
            fh.write(",".join(f"{v:.6f}" for v in pts[k]) + "," + ",".join(f"{v - 70:.6f}" for v in L[ok, k]) + "\n")
    res = compare_to_reference(f, p, pts, str(path), parse_bands(None, f))
    assert abs(res["offset_db"] - 70) < 1e-3 and res["n_spots"] == 40
    for r in res["rows"]:
        assert abs(r["d_median_db"]) < 1e-3 and r["spot_rms_db"] < 1e-3
        assert abs(r["model_p90"] - r["ref_p90"]) < 1e-3


def test_party_rig_reference_file_parses():
    import os
    from spkmod.metrics import load_reference_csv
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "party_rig", "floor_model_deep30_x80.csv")
    pts, centres, L = load_reference_csv(path)
    spots = np.loadtxt(os.path.join(os.path.dirname(path), "floor_spots.csv"), delimiter=",")
    assert pts.shape == spots.shape and np.allclose(pts, spots, atol=1e-3)
    assert L.shape == (len(centres), len(pts)) and 30 < centres[0] < 32


def test_per_frequency_and_diff():
    from spkmod.metrics import diff_rows, per_frequency
    f = log_freqs(40, 100, 12)
    p = np.ones((len(f), 4), complex)
    rows = per_frequency(f, p, {"a": np.arange(4)})
    assert len(rows) == len(f) and all(abs(r["p90"]) < 1e-9 for r in rows)
    d = diff_rows([{"zone": "a", "band": "b", "median_db": 90, "p10": -1, "p90": 1}],
                  [{"zone": "a", "band": "b", "median_db": 92, "p10": -2, "p90": 1}], keys=("median_db", "p10", "p90"))
    assert d == [{"zone": "a", "band": "b", "d_median_db": 2.0, "d_p10": -1.0, "d_p90": 0.0}]


def test_shared_slope_is_not_a_hole():
    """A steep high-pass that every spot shares must not count as a hole; a local dip at one spot must."""
    f = log_freqs(25, 60, 48)
    slope = (f / 40.0) ** 4                       # ~-12 dB across the 25-30 band
    p = np.ones((len(f), 20), complex) * slope[:, None]
    m = band_metrics(f, p, {"25-30": (25, 30), "30-60": (30, 60)}, smoothing=0)
    assert m["25-30"]["holes3_pct"] == 0 and m["25-30"]["holes6_pct"] == 0
    p[np.argmin(abs(f - 45)), 3] *= 0.1           # -20 dB at one spot
    m = band_metrics(f, p, {"30-60": (30, 60)}, smoothing=0)
    assert m["30-60"]["holes6_pct"] == 5.0
