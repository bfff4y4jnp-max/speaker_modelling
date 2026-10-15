import numpy as np

from spkmod import dsp
from spkmod.air import Air, ground_reflection
from spkmod.system import floor_stats, log_freqs


def test_lr4_sums_flat_and_bw_minus3():
    f = log_freqs(20, 20000, 12)
    lp = dsp.crossover(f, 48000, "LinkwitzRileyLowpass", 1000, 4)
    hp = dsp.crossover(f, 48000, "LinkwitzRileyHighpass", 1000, 4)
    assert np.allclose(np.abs(lp + hp), 1, atol=1e-3)
    bw = dsp.crossover(np.array([1000.0]), 48000, "ButterworthLowpass", 1000, 3)
    assert abs(20 * np.log10(abs(bw[0])) + 3.01) < 0.05


def test_camilladsp_pipeline():
    cfg = {"devices": {"samplerate": 48000, "capture": {"channels": 2}},
           "mixers": {"m": {"channels": {"in": 2, "out": 2},
                            "mapping": [{"dest": 0, "sources": [{"channel": 0, "gain": -6}, {"channel": 1, "gain": -6}]},
                                        {"dest": 1, "sources": [{"channel": 0, "gain": 0, "inverted": True}]}]}},
           "filters": {"d": {"type": "Delay", "parameters": {"delay": 1.0, "unit": "ms"}},
                       "pk": {"type": "Biquad", "parameters": {"type": "Peaking", "freq": 100, "q": 2, "gain": 6}}},
           "pipeline": [{"type": "Mixer", "name": "m"}, {"type": "Filter", "channels": [0], "names": ["d", "pk"]}]}
    f = np.array([100.0, 500.0])
    H, fs = dsp.camilladsp_transfer(cfg, f)
    assert abs(20 * np.log10(abs(H[0, 0, 0])) - (6 - 6.02)) < 0.05
    assert np.allclose(H[:, 1, 0], -1)


def test_ground_reflection_limits():
    air = Air()
    q_rigid = ground_reflection(np.array([100.0]), 10.0, 0.3, np.inf, air)
    assert np.allclose(q_rigid, 1)
    q_lf = ground_reflection(np.array([30.0]), 10.0, 0.3, 200e3, air)
    q_hf = ground_reflection(np.array([4000.0]), 10.0, 0.1, 200e3, air)
    assert abs(q_lf[0]) > 0.9 and abs(q_hf[0]) < abs(q_lf[0])


def test_floor_stats_flat_is_zero():
    f = log_freqs(30, 15000, 24)
    p = np.ones((len(f), 20)) * (1 + 0.0j)
    st = floor_stats(f, p)
    for r in st["rows"].values():
        assert abs(r["p10"]) < 1e-9 and abs(r["p90"]) < 1e-9 and r["dips_pct"] == 0
