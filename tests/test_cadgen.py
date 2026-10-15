import os

from spkmod.cadgen import build

HERE = os.path.dirname(__file__)


def test_generic_tapped_horn_spec(tmp_path):
    step, pistons = build(os.path.join(HERE, "..", "examples", "cad_specs", "generic_tapped_horn.yml"), str(tmp_path))
    assert os.path.exists(step) and os.path.exists(tmp_path / "generic_tapped_horn_preview.png")
    assert pistons[0]["center"] == [0.6, 0.25, 0.189] and pistons[0]["cutout"] == 0.282


def test_units_do_not_leak_from_the_mesher(tmp_path):
    """build_scene_mesh sets gmsh's STEP unit to metres; cadgen must still write mm specs at the right size."""
    import gmsh
    import numpy as np
    from spkmod.cadgen import build_spec
    from spkmod.geometry import Cabinet, build_scene_mesh
    spec = {"name": "bx", "units": "mm", "parts": [{"type": "box", "at": [0, 0, 0], "size": [400, 400, 400]}]}
    s1, _ = build_spec(spec, str(tmp_path), name="a", preview=False, quiet=True)
    build_scene_mesh([Cabinet("c", step=s1)], 0.2, ground=False)
    s2, _ = build_spec(spec, str(tmp_path), name="b", preview=False, quiet=True)
    gmsh.initialize()
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    gmsh.model.occ.importShapes(s2)
    gmsh.model.occ.synchronize()
    bb = gmsh.model.getBoundingBox(-1, -1)
    gmsh.finalize()
    assert np.allclose(bb[3:], 0.4, atol=1e-6)


def test_step_hash_ignores_header_and_occ_counter(tmp_path):
    import time
    from spkmod.cadgen import build_spec
    from spkmod.scene import _file_hash
    spec = {"name": "bx", "units": "mm", "parts": [{"type": "box", "at": [0, 0, 0], "size": [300, 200, 100]}]}
    a, _ = build_spec(spec, str(tmp_path / "a"), preview=False, quiet=True)
    time.sleep(1.1)
    b, _ = build_spec(spec, str(tmp_path / "b"), preview=False, quiet=True)
    assert open(a, "rb").read() != open(b, "rb").read() and _file_hash(a) == _file_hash(b)
    spec["parts"][0]["size"][0] = 301
    c, _ = build_spec(spec, str(tmp_path / "c"), preview=False, quiet=True)
    assert _file_hash(c) != _file_hash(a)
