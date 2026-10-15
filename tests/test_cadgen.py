import os

from spkmod.cadgen import build

HERE = os.path.dirname(__file__)


def test_generic_tapped_horn_spec(tmp_path):
    step, pistons = build(os.path.join(HERE, "..", "examples", "cad_specs", "generic_tapped_horn.yml"), str(tmp_path))
    assert os.path.exists(step) and os.path.exists(tmp_path / "generic_tapped_horn_preview.png")
    assert pistons[0]["center"] == [0.6, 0.25, 0.189] and pistons[0]["cutout"] == 0.282
