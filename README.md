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

## Free sharded runs on GitHub Actions

Push a sweep file (any `sweeps/*.yml`), or start **Actions → solve → Run workflow** with its path. The workflow:
1. splits the sweep into shards: each acoustic solve × whole frequency bands, up to 20 at once on free runners (4 cores, 16 GB, 6 h each);
2. solves each shard on its own runner;
3. merges the shards and writes the tables;
4. commits `runs/<name>/` to the same branch: `table.csv` / `.txt`, `perfreq.csv`, `reference.csv`, and `fields/`.

Shards are made of whole bands of the full run, so the merged result matches a single-machine run (tested). Committed fields mean later offline variants, and reruns of the same sweep, only solve what's missing.

Locally, `spkmod sweep S --out runs/<name>` uses the committed fields. The same commands also work on a Slurm cluster:
```bash
spkmod sweep S --out D --shards      # JSON list of shard ids
spkmod sweep S --out D --shard i     # solve one shard into D/parts
spkmod sweep S --out D --merge       # join the parts, write the tables
```
`runs/<name>.yml` requests (`spkmod job plan/solve/merge`) shard a whole project the same way, and `spkmod report --results` reports on the merged results. The runner type is the repo variable `SPKMOD_RUNNER` (default `ubuntu-latest`).

## Sweeps: many variants, one table

```bash
spkmod sweep examples/party_rig/sweeps/sub_layouts.yml --dry-run    # variants and the solves they need
spkmod sweep examples/party_rig/sweeps/sub_layouts.yml --parallel 2 # solve (cached, resumable) + table
```

A sweep file names a base project, the scenes and a list of variants. A variant can do any of these:
* move, add, remove or replace items;
* swap or generate cabinets (`generator: file.py:func` with `params`);
* change wiring, drivers or any project setting;
* make offline-only changes: channel delay, polarity, gain, mute, extra CamillaDSP filters, DSP preset. These reuse an existing solve.

`base:` inherits from another variant and `grid:` expands parameter lists.

Output in `sweeps/<name>/`:
* `table.csv` and `table.txt`: per variant, zone and band, the median level (dB SPL at 0 dBFS), the change against the reference variant, tonal p10/p90 and rms, holes over 6 and 3 dB, and peak excursion;
* the per-channel fields (`fields/*.npz`), so further offline changes need no solve.

Sweeps solve with `solver.interior: reuse` by default. Each cabinet's interior FEM is solved once in its own frame, reduced to its openings and cached, so identical boxes and new layouts only redo the outside BEM. Against the full coupled solve it agrees to within 0.05 dB in driven SPL (`tests/test_reduced.py`). Set `interior: full` to force the full solve.

Comparisons:
```bash
spkmod compare sweeps/sub_layouts --a m2m --b line4                 # difference table B - A per zone and band
spkmod compare sweeps/sub_layouts --variant m2m --reference examples/party_rig/floor_model_deep30_x80.csv
spkmod report project.yml --reference floor_model_deep30_x80.csv    # zones table + floor vs floor model
```
A sweep with `reference_csv:` also writes `reference.csv`, and every sweep writes `perfreq.csv` (zone median and spread at each frequency). The reference is aligned by its median offset, because the floor model's level is arbitrary, and is compared at its own sixth-octave resolution.

Zones come from the project's `zones:`, the sweep's `zones:` and the listener set (`floor`). The metric definitions are in `spkmod/metrics.py`.

Python API:

```python
from spkmod import Cabinet, Piston, solve_scene, Driver, Wiring, solve_network
res = solve_scene([Cabinet("sub", step="sub.step", pistons=[Piston("w", (0.49, .225, .33), (1, 0, 0), 0.258)])],
                  freqs, points=mics, ground=True)
out = solve_network(res, Wiring().add("sub.w", Driver.from_ts("12in", 35, .38, 5, .12, 5.6, .0522), "sub"))
```

See `docs/DESIGN.md` for methods, accuracy, speed, CAD preparation and the roadmap.
