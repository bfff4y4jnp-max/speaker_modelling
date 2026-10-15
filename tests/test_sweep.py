import copy
import os
import shutil

import numpy as np
import pytest
import yaml

from spkmod.sweep import apply_variant, expand_variants

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "examples", "demo")


def test_expand_base_and_grid():
    v = expand_variants([{"name": "base", "offline": {"polarity": {"a": -1}}},
                         {"name": "d", "base": "base", "grid": {"offline.delay_ms.a": [1, 2]}}])
    names = [n for n, _ in v]
    assert names == ["base", "d[a=1]", "d[a=2]"]
    assert v[2][1]["offline"] == {"polarity": {"a": -1}, "delay_ms": {"a": 2}}


def test_apply_variant_geometry_ops():
    with open(os.path.join(DEMO, "project.yml")) as fh:
        cfg = yaml.safe_load(fh)
    v = {"move": {"sub1": {"position": [1, 2, 0]}},
         "add": [{"cabinet": "sub", "name": "sub2", "position": [0, 1, 0],
                  "wiring": {"woofer": {"driver": "woofer12", "channel": "sub"}}}],
         "project": {"solver": {"h_max": 0.2}}}
    out = apply_variant(cfg, v, ["bass"], DEMO, "/tmp", DEMO)
    items = out["scenes"]["bass"]["items"]
    assert items[0]["position"] == [1, 2, 0] and items[1]["name"] == "sub2"
    assert "sub2.woofer" in out["wiring"]["bass"] and "tops" not in out["scenes"]
    assert out["solver"]["h_max"] == 0.2 and cfg["solver"]["h_max"] == 0.10   # input untouched
    out2 = apply_variant(out, {"remove": ["sub2"]}, ["bass"], DEMO, "/tmp", DEMO)
    assert len(out2["scenes"]["bass"]["items"]) == 1 and "sub2.woofer" not in out2["wiring"]["bass"]


GEN = '''
def box(w=0.4, wall=18):
    W = w * 1000
    return {"name": "genbox", "units": "mm",
            "parts": [{"type": "box", "at": [0, 0, 0], "size": [W, 400, 400]}],
            "cutouts": [{"type": "box", "at": [wall, wall, wall], "size": [W - 2 * wall, 400 - 2 * wall, 400 - 2 * wall]},
                        {"type": "cylinder", "center": [W - wall / 2, 200, 200], "axis": [1, 0, 0], "diameter": 220, "depth": 40}],
            "pistons": [{"name": "woofer", "center": [W - wall / 2, 200, 200], "axis": [1, 0, 0], "sd_cm2": 300, "cutout": 220}]}
'''


def test_sweep_end_to_end(tmp_path):
    """Demo bass scene at two frequencies: geometry, generator and offline variants, one table."""
    from spkmod.sweep import run_sweep
    d = tmp_path / "demo"
    shutil.copytree(DEMO, d, ignore=shutil.ignore_patterns("report", ".spkmod_cache"))
    (d / "gen.py").write_text(GEN)
    spec = {"project": "project.yml", "scenes": ["bass"], "frequencies": {"list": [40.0, 60.0]},
            "listeners": {"points": [[3, 0, 1.6], [4, 1, 1.6], [4, -1, 1.6]]},
            "zones": {"behind": {"points": [[-3, 0, 1.6]]}}, "bands": {"35-65": [35, 65]}, "smoothing": 0,
            "variants": [{"name": "base"},
                         {"name": "moved", "move": {"sub1": {"position": [0.5, -0.25, 0]}}},
                         {"name": "pol", "base": "base", "grid": {"offline.polarity.sub": [1, -1]}},
                         {"name": "gen", "cabinets": {"sub": {"generator": "gen.py:box", "params": {"w": 0.45}}}}]}
    (d / "sweep.yml").write_text(yaml.safe_dump(spec))
    rows = run_sweep(str(d / "sweep.yml"))
    names = sorted({r["variant"] for r in rows})
    assert names == ["base", "gen", "moved", "pol[sub=-1]", "pol[sub=1]"]
    get = lambda v, z: next(r for r in rows if r["variant"] == v and r["zone"] == z)
    # offline polarity flip of the only channel: same level; identical to base when +1
    assert abs(get("pol[sub=1]", "floor")["median_db"] - get("base", "floor")["median_db"]) < 1e-6
    assert abs(get("pol[sub=-1]", "floor")["median_db"] - get("base", "floor")["median_db"]) < 1e-6
    assert get("moved", "floor")["d_median_db"] != 0
    out = d / "sweeps" / "sweep"
    assert (out / "table.csv").exists() and len(list((out / "fields").glob("*.npz"))) == 3
    assert (out / "perfreq.csv").exists()
    from spkmod.sweep import load_variant
    v = load_variant(str(out / "variants" / "moved.npz"))
    assert list(v["zones"]) == ["floor", "behind"] and v["p"].shape == (2, 4)
    from spkmod.cli import main
    main(["compare", str(out), "--a", "base", "--b", "moved", "--out", str(out / "diff.csv")])
    assert (out / "diff.csv").read_text().startswith("zone,band,d_median_db")
    # rerun: everything cached, no new solves
    rows2 = run_sweep(str(d / "sweep.yml"))
    assert rows2 == rows


def test_sharded_sweep_matches_single_run(tmp_path):
    """Shards (solve group x whole frequency bands) solved separately and merged give the same table."""
    from spkmod import sweep as sw
    from spkmod.system import log_freqs
    d = tmp_path / "demo"
    shutil.copytree(DEMO, d, ignore=shutil.ignore_patterns("report", ".spkmod_cache"))
    spec = {"project": "project.yml", "scenes": ["bass"], "frequencies": {"list": log_freqs(35, 150, 4).tolist()},
            "listeners": {"points": [[3, 0, 1.6], [4, 1, 1.6], [4, -1, 1.6]]}, "bands": {"35-150": [35, 150]},
            "variants": [{"name": "base"}, {"name": "moved", "move": {"sub1": {"position": [0.5, -0.25, 0]}}},
                         {"name": "pol", "base": "base", "offline": {"polarity": {"sub": -1}}}]}
    (d / "sweep.yml").write_text(yaml.safe_dump(spec))
    single = sw.run_sweep(str(d / "sweep.yml"), out=str(d / "single"))
    ctx = sw.prepare(str(d / "sweep.yml"), out=str(d / "sharded"), quiet=True)
    sh = sw.shards(ctx)
    assert len(sh) == 4 and {s["group"] for s in sh} == set(ctx["groups"])
    for s in sh:                                    # each would run on its own runner
        sw.solve_shard(sw.prepare(str(d / "sweep.yml"), out=str(d / "sharded"), quiet=True), s["shard"])
    assert len(sw.merge_shards(ctx)) == 2
    sharded = sw.tables(sw.prepare(str(d / "sweep.yml"), out=str(d / "sharded"), quiet=True))
    assert [r["variant"] for r in sharded] == [r["variant"] for r in single]
    for a, b in zip(single, sharded):
        assert abs(a["median_db"] - b["median_db"]) < 1e-3 and abs(a["p90"] - b["p90"]) < 1e-3
    assert sw.shards(sw.prepare(str(d / "sweep.yml"), out=str(d / "sharded"), quiet=True)) == []   # nothing left to solve
