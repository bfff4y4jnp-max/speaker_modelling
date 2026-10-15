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
