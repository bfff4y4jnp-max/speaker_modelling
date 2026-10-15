# spkmod: design

3D FEM/BEM modelling of loudspeakers and sound systems from CAD, built as the
follow-on to the Outdoor Party Rig work. Its first job is to replace the
assumptions in the existing floor model (1D horn models, idealised box
directivity, ground reflections added by hand) with full-wave 3D results for
the actual cabinets. The rig being modelled has tapped-horn subs, a folded
low-mid horn, the pole box and the fills.

## 1. How this is normally done, and what spkmod changes

| Approach | Typical tools | Strengths | Weaknesses | spkmod |
|---|---|---|---|---|
| Lumped Thiele-Small | WinISD, BassBox | Instant, good below the first box mode | No geometry, no directivity, no horn path | Kept as the driver model. Air load and box loading come from the 3D solve, so nothing is double counted (`Mmd` not `Mms`) |
| 1D transmission line / horn T-matrix | Hornresp, AkAbak | Fast, good for straight horns and lines | Folds, bends, mouth-to-mouth and stacking effects are guessed; directivity is idealised | Full 3D FEM of the folded path, with the same viscothermal losses (validated against Kirchhoff) |
| Full 3D FEM with PML | COMSOL | General | Large air domain, slow, licence | FEM only inside the cabinet envelope. BEM outside, so there is no exterior air mesh and no PML |
| BEM (exterior and interior) | ABEC, VACS | Exact radiation, polars | Interior folds need many BEM elements. Commercial (ABEC is free but Windows-only and scripted by hand) | Coupled FEM inside + BEM outside. Python, scriptable, cross-platform, cached |
| Point-source / balloon system sims | Mapp, EASE, the current floor model | Fast whole-system coverage | Far-field balloons are wrong in the near field and between stacked boxes. Ground and boundary coupling are approximate | Listener points are evaluated **exactly** (near field included) from the BEM surface solution. Interacting boxes are solved jointly |
| Measurement-based | VituixCAD, REW | Real drivers | Needs built boxes and good outdoor measurements | Inputs accept measured T/S. The roadmap includes fitting to measured impedance |

The main improvements are:

1. **The FEM/BEM interface is the cabinet's envelope (bounding box).** The
   BEM only sees a few simple closed surfaces, whatever the internal
   folding. Recesses, odd shapes and open-backed panels are handled
   exactly by the FEM. Thin-body problems in the BEM are avoided entirely.
2. **Joint scenes.** The four tapped horns, mouth to mouth across a 1.2 m
   gap, plus the low-mid on rigid ground, are one acoustic problem. Mutual
   radiation impedance between drivers in different boxes is included. Cardioid
   rejection and floor coupling are then predicted rather than assumed.
3. **Electro-acoustic separation.** Each scene is solved once per frequency
   for unit velocity of each vibrating body. Drivers, wiring, polarity,
   CamillaDSP filters, delays, presets and amp limits are applied afterwards
   in milliseconds. Iterating DSP never re-runs the acoustics.
4. **The actual DSP.** The CamillaDSP YAML running on the Pi (biquads,
   LR/BW crossovers, delays, gains, mixers, FIR from file) is evaluated
   exactly in the z-domain at the analysis frequencies.

## 2. Architecture

```
STEP (solids)  +  project.yml (pistons, stuffing, placement, wiring, DSP, listeners)
      │
      ▼
geometry.py   gmsh/OCC: fragment solids, pistons, envelope, stuffing → classify
      │       air | solid | piston; faces → aperture / rigid / vibrating / wall
      ▼
scene.py      per frequency band: mesh (h ≤ λ/6), FEM (fem.py) + BEM (bem.py)
      │       coupled solve per body DOF → Za (radiation/load impedance matrix),
      │       T (pressure at listener points), T_far (balloons), U_ap (port/mouth flow)
      │       → cached .npz (keyed by geometry, settings, frequencies)
      ▼
drivers.py    T/S + LR-2 voice coil, passive radiators, wiring/polarity
      │       (Zm + Za + Bl²/Ze) v = Bl e/Ze   → velocities, excursion, current, |Z|
      ▼
system.py     sum of scenes × channel transfer (dsp.py: CamillaDSP) × amp gain,
              grass ground (Miki impedance, spherical-wave Q) for elevated scenes,
              ISO 9613-1 air absorption, floor statistics, max-SPL limits
```

Hierarchy: **Driver → Cabinet → Scene → System.**

* Driver: an electro-mechanical model. Cone driver, passive radiator, and later a DML plate or exciter.
* Cabinet: one CAD part with pistons and stuffing.
* Scene: cabinets that must be solved together (they couple acoustically), with or without ground.
* System: scenes that couple only through the listener. Different scenes can cover different frequency ranges.

## 3. Numerical methods

**Conventions.** Solvers use e^{-iωt} internally and outputs are conjugated to the
engineering e^{+jωt} convention. All phasors are rms. SI units throughout. YAML
keys with unit suffixes (`_mm`, `_cm2`, `_l`, `_mH`, `_g`) are converted.

**Interior FEM** (`fem.py`, scikit-fem):
* Quadratic tetrahedra; the Helmholtz equation is solved per air region.
* Fibre-filled regions use the Miki equivalent fluid, set by flow resistivity.
* Rigid walls carry the Berggren–Bernland–Noreland viscothermal boundary-layer condition. This is what sets the Q of horn and port resonances, and therefore the ringing.
* Facet couplings are exact: for P2 on a triangle, the vertex basis functions integrate to 0 and the edge functions to area/3.

**Exterior BEM** (`bem.py`, bempp-cl):
* Formulation: Burton–Miller with η = i/k, Galerkin P1 pressure and DP0 normal velocity on active faces only. This is free of the irregular frequencies that plague plain BEM of closed boxes.
* Rigid ground: the image method. The scene is mirrored, assembled, and folded back to the original unknowns, so the LU is half size. Faces on z = 0 are dropped, so boxes standing on the ground are exact.
* Field points: vectorised quadrature of the representation formula. It is valid anywhere outside the bodies, including the near field.

**Coupling.**
* Pressure continuity is weak, per aperture triangle; velocity continuity is strong.
* One sparse direct factorisation per frequency solves all body DOFs at once.

**Driver.**
* Moving mass: Mmd = Mms − 2·(8/3)ρa³ (Small's convention) unless Mmd is given, because the solve supplies the air load.
* Voice coil: Le plus the optional LR-2 (R2 ∥ L2) inductance model.
* Round diaphragms: meshed as polygons, then rescaled to the exact Sd so volume velocity is exact.

**Ground for elevated scenes.**
* The scene is solved free-field and also evaluated at mirrored listener points.
* The image contribution is multiplied by the spherical-wave reflection coefficient Q (Miki surface impedance, boundary-loss function F(w)) for the chosen flow resistivity. 200 kPa·s/m² is short grass.
* Alternatively, set a scene's `ground_reflection: [[500, 1.0], [4000, 0.5]]` to use a tabulated real |R|(f), like the floor model's grass curve. It is interpolated in log f, held constant outside the table, and angle independent. It takes precedence over `ground_sigma`.
* On-ground scenes (subs, low-mid) use the rigid BEM ground. Grass is effectively rigid below 300 Hz.

## 4. Speed

| Technique | Gain |
|---|---|
| Frequency bands, one mesh each, h = λ/6 at the band top (capped by `h_max`). Bands with equal h share a mesh | No over-meshing at low frequency |
| **Frequency interpolation of BEM matrices**: assemble at 3–9 Chebyshev wavenumbers per band, interpolate A(k)e^{−ikr} entrywise | ~10× fewer assemblies; verified error ~1e-8 |
| Quadrature order 3 (vs bempp default 4) | ~1.4× faster, same accuracy (checked) |
| OpenCL CPU backend auto-selected (`pip install spkmod[fast]`) | 2.3× faster assembly, 5× less JIT warm-up |
| Image folding for ground | LU on half the unknowns |
| Multi-RHS: one factorisation per frequency for all drivers and apertures | Drivers are nearly free |
| **Reused interiors** (`solver.interior: reuse`): each cabinet's interior FEM is reduced once to an operator on its openings. Layouts then couple it to the scene BEM through exact overlap areas | Identical boxes share one interior solve, and moving boxes redoes only the outside BEM. Within 0.05 dB driven SPL of the full solve |
| Result cache + post-processing on stored transfer functions | DSP, presets and wiring changes are instant |
| Memory budget (60% of free RAM, or `solver.memory_gb`) | The mesh coarsens with a warning instead of swapping. The same project runs on a 16 GB laptop or in the cloud |

Rough cost on 4 cores:

* A single box to 600 Hz: about 1–2 minutes, including about 20 s of first-run JIT.
* The ground stack to 500 Hz: an estimated 5–15k BEM triangles after mirroring, so 10–40 minutes once, then cached.
* The pole box to 3 kHz: minutes.
* The pole box to 16 kHz: hours. It needs a ~3 mm mesh on the waveguide, so run it as a separate scene with a reduced `h_max` only when needed.

## 5. Accuracy: validation suite (`pytest`)

| Test | Reference | Result |
|---|---|---|
| Pulsating sphere, near and far field | Analytic | < 2% (mesh-limited) |
| Pulsating hemisphere on rigid ground (image folding) | Analytic sphere | < 2% |
| Burton–Miller at a sphere's interior resonance ka = π | Analytic | 0.01% (plain BEM: condition 180 → 5) |
| Closed tube with wall losses, including at resonance | Kirchhoff wide-tube theory | < 3% |
| Sealed box: air-spring stiffness | ρc²Sd²/V | 0.5% |
| Sealed box: LF far field | Monopole | < 1.5% |
| Vented box tuning (FEM–BEM aperture coupling) | Helmholtz with end corrections | Within the end-correction range (30.6 Hz vs 30.0–30.7 Hz) |
| LR4 sum, BW −3 dB, CamillaDSP mixer and filter chain | Theory | Exact |
| Ground Q limits, floor-stat sanity | Theory | Pass |

What still limits real-world accuracy, roughly in order:

* **Driver parameters.** Datasheet T/S values are often ±10–20%. Measure the impedance of each built box and fit; this is on the roadmap.
* **Rigid walls.** Plywood panels are assumed rigid; panel resonances are on the roadmap.
* **Linear small signal.** No Bl(x) or Cms(x), and no thermal compression.
* **Crowd absorption** is not modelled. Use the floor model's crowd factor or a lossy layer.
* **Weather.** Wind and temperature gradients are ignored; ISO 9613-1 air absorption at the set temperature and humidity is included.

## 6. Preparing CAD (Fusion / FreeCAD / SolidWorks → STEP)

* Export only the **solid parts**: panels, braces, port tubes, the waveguide body. Merging into one body is fine but not required; overlapping solids are fragmented automatically.
* Do **not** model the air or the drivers. Leave the driver cut-outs open, and place each driver in `project.yml` as a piston:
  * centre and axis (direction of forward cone motion) in the CAD's coordinates, in metres;
  * Sd;
  * cut-out diameter (the frame fills the hole around the piston);
  * optional motor diameter and depth, which displaces air in rear chambers and horn paths.
* Units are read from the STEP file (mm is typical) and converted to metres.
* By default the envelope is the bounding box of the solids. The FEM fills everything inside it that isn't solid, and the outer faces of that air become the mouths and ports automatically. For very non-box shapes you may give an explicit `envelope` box.
* Check before solving with `spkmod mesh project.yml --scene stack --out stack.vtk` (open it in ParaView). `kind` 0 = rigid, 1 = aperture, 2 = vibrating.

## 7. Using it for the party rig

1. Export STEP for `fold3_35`, `lowmid`, `polebox` (with the waveguide) and `fill`. Fill in `examples/party_rig/project.yml`.
2. Put the driver data in. Datasheet values are fine to start. For better accuracy, measure the impedance of the built boxes and fit Re, Le, Bl, Mms, Cms, Rms.
3. `spkmod solve project.yml --scene stack --per-octave 12` gives a quick look. Use 24 points per octave for the final run (the horn resonances are sharp).
4. Copy the Pi's CamillaDSP config and FIR files to `dsp/`, then run `spkmod report project.yml --out report`. The output includes:
   * floor statistics in the same bands as the floor model (p10/p90 spread, % of spots with dips over 6 dB);
   * coverage map, impedance, excursion and polars.
5. Cross-check against the existing floor model, band by band. Where they disagree, the 3D model is the one with fewer assumptions. The likely places are cardioid depth behind the stack, the 180–450 Hz handover, and stacking near the ground.

## 8. Roadmap

* **Distributed Mode Loudspeakers (designed for, not yet built).** The acoustic solver already works in generalised body DOFs: velocity shape per face × DOF, and returns the full radiation impedance matrix. A DML is:
  * a body whose DOFs are plate modes, from an orthotropic Kirchhoff/Mindlin plate FEM (scikit-fem Morley/Argyris elements) with loss factor, edge suspension stiffness/damping and exciter coil masses;
  * plus one DOF per inertial exciter's magnet.

  The exciter is a `Driver` whose coil couples to the mode shapes at its attachment point. Open-backed panels need nothing new: the panel sits inside its envelope box and the FEM air surrounds both faces. Work needed: `PlateBody` (modal FEM, ~200 lines), a `shapes()` hook in place of the piston-only shape, and validation against a baffled simply-supported plate (Rayleigh integral).
* Fit drivers to measured impedance curves (`.zma`). Import measured balloons for parts not modelled.
* Large-signal limits per track, matching the existing `twoview.py`: excursion and voltage headroom are already exposed (`SystemModel.max_input`).
* Panel vibration (wood FEM shell), cone breakup (axisymmetric shell), Bl(x)/Cms(x).
* Automatic optimisation of delays, EQ and polarity for floor-consistency metrics.
* Impulse and group-delay views via IFFT, to quantify the tapped-horn ringing and the FIR alignment.
* FMM (exafmm) or H-matrix compression for scenes above ~20k BEM elements, mostly full-range waveguides.
