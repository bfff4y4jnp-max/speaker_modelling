# Notes for the spkmod code (from the design project, 2026-10-15)

These are requests and findings for whoever works on the solver code. The design project only writes inputs and runs and checks the solves. It does not change `spkmod/`.

## Found while building the party-rig inputs

1. **Default mesh is too coarse for the horns.** `project.yml` uses `h_max: 0.15`. The folds have narrow gaps:
   - 25 mm front gap in the deep30, fold3_35 and deep27 subs;
   - 8 mm front chamber and 40 mm slot in the low-mid.

   Please refine near narrow gaps automatically (local size from the distance between facing walls), or warn when a gap is narrower than about h/3.

   One deep30 horn at h 0.15 put the impedance peaks 8–10 % higher than the 1D model (46/86/123 Hz against 42/79/103 Hz). It also nearly removed the 1D null at ~110 Hz. A run at h 0.05 is checking whether that is the mesh.
2. **Piston inside a curved throat fails to mesh.** The DE250 piston sat at the throat of the lofted waveguide bore with `cutout` equal to the hole. `gmsh` then failed with "PLC Error: A segment and a facet intersect".
   - Workaround in `polebox.yml`: the piston sits 5 mm behind the throat in a 32 mm straight hole, with a frame (`cutout`) of 36 mm.
   - A 1 mm step between the motor's front face and a nearby solid face also failed, with "1D mesh not a closed loop".
   - A clearer error naming the piston would help, or snapping coincident faces.
3. **Thin curved shells** (a 6 mm printed waveguide wall with air on both sides) gave the same PLC error at practical h. The waveguide is therefore modelled as a solid plug with the bore cut out. That is fine for radiation, since the air around the shell is sealed bay air.
4. **`Driver(**d)` rejects unknown keys.** A `source:` field in `drivers:` raises a TypeError, so driver sources are YAML comments for now. Consider allowing `source`/`notes` keys.
5. **Ground in the pole and fills scenes.** These use `ground_sigma: 200000` (impedance ground). The floor model's grass is an |R| curve instead: 1 below 500 Hz, falling to 0.5 at 4 kHz. For like-for-like checks, an option to use a tabulated |R|(f) would help.

## Reusable notes from the project's earlier BEM thread (stopped 2026-10-15)

Its files are in the design project at `/mnt/project-files/sim/` (`HANDOVER.md`, `bemcore.py`, `validate.py`, `geom_lowmid.py`, `geom_wg.py`).

- **Validation** with bempp-cl 0.4.2:
  - Burton-Miller pulsating sphere ≤ 0.05 dB for ka 0.5–4.49, including ka = π.
  - Piston cap on a sphere against the Morse series: directivity ≤ 0.015 dB, radiation load ≤ 0.4 % for ka 0.5–3.
  - These are useful extra cases for `tests/test_bem.py`.
- **Speed** (4 cores, dense, numba):
  - First call per operator and space pair: about 15 s of JIT.
  - 4.1k nodes with Burton-Miller: about 150 s per frequency.
  - Above about 7k nodes it needs FMM.
- **FMM:** exafmm-t has no pip wheel. Clone github.com/exafmm/exafmm-t, apt `libfftw3-dev libopenblas-dev`, run `python3 setup.py build_ext`, then copy `build/lib*/exafmm*.so` into site-packages. It is untested with spkmod.
- **numba** fails with FileNotFoundError when the working directory is on a network mount. Run from a local directory.
- **Waveguide stand-in:** `geom_wg.py` builds an OS 120 × 70 waveguide as a spline loft:
  - 25.4 mm throat, 10° throat angle, 100 mm deep;
  - morphs to a 320 × 245 mouth.

  At h 6/20 mm it is 17k nodes. It is closer to the printed profile than the elliptical bore used here, so it could replace `waveguide_face.step` once meshing a thin shell works.
- **Checks it suggested:**
  - the low-mid fold against the 1D model (throat load, on-ground SPL, floor map; the 1D model expects 1–2 dB differences at 250–400 Hz);
  - the waveguide's real −6 dB beamwidth against the 60°/35° Gaussian the floor models assume;
  - the sub mouths on grass.
