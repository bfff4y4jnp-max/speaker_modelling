# Party rig: current build as spkmod inputs

The handover's six deliverables, generated from the Subwoofer Design project's own design files.

**Build:**
- Subs: 4 × deep30 tapped horns, 2+2 mouth to mouth, with the rear row inverted and delayed 3.50 ms.
- Low-mid: the straight-cut horn with 2 × 12-280-W, crossing over to the subs at 80 Hz.
- Pole box: 2 × 8FE200 (4 Ω) and a DE250 on a 120 × 70 waveguide.
- Fills: M/S, 2 × PC105-4 each.
- DSP: the integration thread's `_deep70` presets.

| File | What it is |
|---|---|
| `cad/specs/deep30.yml`, `fold3_35.yml`, `deep27.yml` | Tapped-horn subs, built from each horn's `fold_params` (straight-cut 3-layer fold) |
| `cad/specs/lowmid.yml` | Low-mid folded horn, from the build section drawing |
| `cad/specs/polebox.yml` + `cad/waveguide_face.step` | Pole box, from the cut list. The waveguide is simplified (see the spec header) |
| `cad/specs/fill.yml` | Stereo fill, 140 × 560 × 180 mm sealed |
| `project.yml` | Drivers, cabinets, layout, wiring, DSP and listeners, all in one file (it holds `drivers.yml` and `layout.yml`) |
| `floor_spots.csv` | The floor model's 335 spots × 3 ear heights, in spkmod's frame |
| `floor_model_deep30_x80.csv`, `floor_model_deep30_x80_bands.txt` | Floor-model cross-check per spot and sixth-octave band, plus its band table |
| `dsp/` | CamillaDSP day, party and peak presets, plus the 16k-tap FIR |
| `make_specs.py`, `make_project.py` | Generators: edit these, not the outputs |

Each spec starts with the §3.7 self-check as comments. `spkmod cad` gives these net air volumes against the design:

| Cabinet | spkmod net air | Design |
|---|---|---|
| deep30 | 398 L | 396 L |
| fold3_35 | 285 L | 283 L |
| deep27 | 389 L | 384 L |

All three scenes mesh with every driver found.

**Guesses (flagged in the files):**
- Driver motor sizes.
- The PC105-4 cut-out.
- The DE250 parameters. They are placeholders: the floor model only used its sensitivity, so its level here is not meaningful, but its coverage is.
- The waveguide's exact shape.
- Grass as σ = 200 kPa·s/m². The floor model used |R| = 1 below 500 Hz, falling to 0.5 at 4 kHz.

```bash
python examples/party_rig/make_specs.py /mnt/project-files/outdoor_sub    # specs + waveguide STEP
spkmod cad examples/party_rig/cad/specs/*.yml --out examples/party_rig/cad
python examples/party_rig/make_project.py /mnt/project-files              # project.yml, CSVs, DSP copies
cd examples/party_rig && spkmod mesh project.yml --scene stack --out stack.vtk
```
