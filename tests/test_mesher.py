"""Mesher robustness (mesh only, no solving)."""
import dataclasses
import logging
import os

import numpy as np
import pytest

from spkmod.geometry import Cabinet, MeshError, Piston, build_scene_mesh

HERE = os.path.dirname(os.path.abspath(__file__))
RIG = os.path.join(HERE, "..", "examples", "party_rig", "project.yml")


def _pole_with_throat_piston():
    from spkmod.project import Project
    cab = Project(RIG).cabinets("pole")[0]
    ax = np.array([0.7314, 0.0, -0.682])
    hf = dataclasses.replace(cab.pistons[2], center=tuple(np.array([0.2246, 0.18, 0.7485]) + 0.005 * ax), cutout=0.0254)
    return dataclasses.replace(cab, pistons=cab.pistons[:2] + [hf])


def test_piston_at_curved_throat_is_nudged_or_named(caplog):
    """The design project's failing DE250 placement: frame circle on the throat edge of the lofted bore."""
    cab = _pole_with_throat_piston()
    with pytest.raises(MeshError, match=r"pole\.hf.*frame ring"):
        build_scene_mesh([cab], 0.15, ground=False, auto_fix=False)
    with caplog.at_level(logging.WARNING, logger="spkmod"):
        m = build_scene_mesh([cab], 0.15, ground=False)
    assert len(m.body_names) == 3 and len(m.tets)
    assert any("pole.hf" in r.message and "mm along its axis" in r.message for r in caplog.records)


def _gap_box(gap):
    def build(occ):
        o = occ.addBox(0, 0, 0, 0.6, 0.4, 0.4)
        i = occ.addBox(0.018, 0.018, 0.018, 0.564, 0.364, 0.364)
        bx, _ = occ.cut([(3, o)], [(3, i)])
        div = occ.addBox(0.018, 0.018, 0.018 + gap, 0.45, 0.364, 0.018)     # shelf `gap` above the floor
        return [t for d, t in bx if d == 3] + [div]
    return build


def test_narrow_gap_reported_not_panels_or_corners(caplog):
    cab = Cabinet("g", builder=_gap_box(0.02), pistons=[Piston("d", (0.591, 0.2, 0.25), (1, 0, 0), 0.1, cutout=0.12)])
    with caplog.at_level(logging.WARNING, logger="spkmod"):
        build_scene_mesh([cab], 0.15, ground=False)
    msg = " ".join(r.message for r in caplog.records if "narrow air gaps" in r.message)
    assert "g: 20 mm" in msg
    assert "18 mm" not in msg            # panel thickness (solid between faces) is not a gap
    with caplog.at_level(logging.WARNING, logger="spkmod"):
        fine = build_scene_mesh([cab], 0.15, ground=False, refine_gaps=True)
        coarse = build_scene_mesh([cab], 0.15, ground=False)
    assert len(fine.tets) > len(coarse.tets)


def test_thin_curved_shell_with_air_both_sides():
    def shell(occ):
        o = occ.addCone(0.05, 0.15, 0.15, 0.10, 0, 0, 0.0127 + 0.006, 0.10 + 0.006)
        i = occ.addCone(0.05, 0.15, 0.15, 0.10, 0, 0, 0.0127, 0.10)
        sh, _ = occ.cut([(3, o)], [(3, i)])
        b = occ.addBox(0, 0, 0, 0.012, 0.30, 0.30)
        h = occ.addCylinder(-0.01, 0.15, 0.15, 0.04, 0, 0, 0.0127)
        bp, _ = occ.cut([(3, b)], [(3, h)])
        return [t for d, t in sh + bp if d == 3]
    cab = Cabinet("wg", builder=shell, pistons=[Piston("hf", (0.006, 0.15, 0.15), (1, 0, 0), 0.0254, cutout=0.0254, thickness=0.004)])
    for h in (0.05, 0.15):
        assert len(build_scene_mesh([cab], h, ground=False).tets)
