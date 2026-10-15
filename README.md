# speaker_modelling: `spkmod`

3D FEM/BEM modelling of loudspeakers and complete sound systems from CAD (STEP).
Coupled interior FEM (scikit-fem) and exterior BEM (bempp-cl) solve the cabinets.
Thiele-Small drivers, CamillaDSP filters, amps, a grass ground and a listener grid
are layered on top. It is the 3D follow-on to the Outdoor Party Rig floor model.

* Models individual **drivers** (impedance, excursion, current), **cabinets**
  (horn/port/box loading, polars, balloons) and the **whole system**
  (coverage, floor consistency, cardioid rejection, max SPL).
* Validated against analytic solutions (see `docs/DESIGN.md` §5).
* Fits the machine: it auto-sizes to available RAM (16 GB laptop or cloud).
  Quick runs use a limited frequency range; full range is slower.
* Built so Distributed Mode Loudspeakers can be added (roadmap in the design doc).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[fast,test]"      # [fast] = OpenCL CPU backend, 2-5x faster BEM
# Linux without a display may need: apt install libglu1-mesa libxcursor1 libxinerama1 libxft2
pytest -q                          # validation suite, ~2 min
```

## Use

```bash
python examples/demo/make_cad.py                         # writes demo STEP files
spkmod mesh   examples/demo/project.yml --scene bass --out bass.vtk   # inspect tagging in ParaView
spkmod solve  examples/demo/project.yml                  # cached in .spkmod_cache
spkmod report examples/demo/project.yml --out report     # impedance, excursion, polars, floor stats, coverage
```

Handover for the design project (creating cabinet specs without CAD software): `docs/HANDOVER.md`.
`spkmod cad spec.yml` turns a panel-list YAML into STEP plus a preview PNG.

The party-rig template is `examples/party_rig/project.yml`. Fill in its TODOs
(STEP files, driver data, layout, CamillaDSP config).

## Free parallel runs on GitHub Actions

Write a run request and push it; every scene and frequency band then solves on its own runner (about 20 in parallel, free for public repos):

```yaml
# runs/stack_v1.yml
project: examples/party_rig/project.yml
scenes: [stack]          # optional
per_octave: 24           # optional
max_jobs: 20
```

The `solve` workflow commits the merged results, `run_info.json` and the report to `runs/stack_v1/`. The summary table also shows on the run page.
Each job is limited to 4 cores, 16 GB and 6 h. The same `spkmod job plan/solve/merge` commands work on Slurm (Iridis) job arrays.

Python API:

```python
from spkmod import Cabinet, Piston, solve_scene, Driver, Wiring, solve_network
res = solve_scene([Cabinet("sub", step="sub.step", pistons=[Piston("w", (0.49, .225, .33), (1, 0, 0), 0.258)])],
                  freqs, points=mics, ground=True)
out = solve_network(res, Wiring().add("sub.w", Driver.from_ts("12in", 35, .38, 5, .12, 5.6, .0522), "sub"))
```

See `docs/DESIGN.md` for methods, accuracy, speed, CAD preparation and the roadmap.
