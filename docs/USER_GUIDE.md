# spkmod USER GUIDE

**Start here.** This is the reference for running models with this repo: what to write, which command to run, and what comes back. Methods and validation are in [`DESIGN.md`](DESIGN.md). The tooling brief is in [`BRIEF_tooling.md`](BRIEF_tooling.md).

Contents:
1. Quick start
2. Running on GitHub (free, no setup)
3. Project file reference (`project.yml`)
4. Cabinet specs (panel lists → STEP)
5. Sweeps (variants → one table)
6. Outputs and metric definitions
7. Conventions and known limits
8. Local install and commands

---

## 1. Quick start

The fastest way to run a study:

1. Write a sweep file next to the project: `examples/party_rig/sweeps/<name>.yml` (format in §5).
2. Commit and push it, on any branch.
3. About 5–10 minutes later, the results are committed to `runs/<name>/` on the same branch: `table.txt`, `table.csv`, `perfreq.csv`, and `reference.csv` if you set `reference_csv:`.

Nothing has to be installed and nothing costs money; see §2. To check a sweep before solving, run `spkmod sweep <file> --dry-run` locally. It lists the variants and how many solves they need.

## 2. Running on GitHub (free, no setup)

The workflow `.github/workflows/solve.yml` starts when any of these happens:
* a push that adds or changes a file matching `**/sweeps/*.yml` (a sweep) or `runs/*.yml` (a whole-project request, §8);
* **Actions → solve → Run workflow**, with `request` set to the file path (e.g. `examples/party_rig/sweeps/sub_layouts.yml`).

What it does:
1. **plan:** lists the variants and splits the work into shards (acoustic solve × whole frequency band), at most 20.
2. **solve:** one free runner per shard (4 cores, 16 GB, 6 h limit). Each shard's run time is shown in the job summary.
3. **merge:** joins the shards, writes the tables, shows `table.txt` on the run page, and commits `runs/<name>/` to the same branch. Fields are committed too, so a later edit that only adds offline variants (delay, polarity, preset) needs no new solve.

Rules of thumb:
* **One sweep file per push.** If several change in one push, only the first is solved (with a warning); start the others with Run workflow.
* **Never edit `runs/<name>/` by hand.** To start over, delete `runs/<name>/fields/`.
* **If any variant ends up with no solved fields, the merge fails loudly.** The run goes red, `table.txt` starts with an `!! INCOMPLETE` line naming the variants, and the partial table is kept as the run's `results-<name>` artifact. Nothing is committed.
* **Runner type** is the repo variable `SPKMOD_RUNNER` (default `ubuntu-latest`). Larger paid runners work unchanged.
* **Run time:** the `sub_layouts` example (8 variants, 4 solves, 67 frequencies, 1096 listener points) runs as 12 shards of about 2–3 minutes each.

## 3. Project file reference (`project.yml`)

Lengths are in **metres** and angles in **degrees**, unless a key carries a unit suffix (`_mm`, `_cm2`, `_l`, `_mH`, `_g`, `_mm_per_N`). Paths are relative to the project file.

The coordinate frame is chosen by the project. For the party rig it is x forward onto the floor, y left, z up, with the ground at z = 0.

```yaml
air: {temperature: 18, humidity: 70}          # also pressure (Pa)

solver:
  elements_per_wavelength: 6    # mesh h = c / (f_top * this), per octave band
  h_max: 0.15                   # never coarser than this (m): resolves the geometry
  h_min: null                   # default h/8
  band_octaves: 1.0             # one mesh per band of this width
  interior: full                # full | reuse (each cabinet's interior solved once: layouts only redo the exterior; sweeps default to reuse)
  refine_gaps: false            # true: mesh reported narrow air gaps finer (they are always reported)
  losses: true                  # viscothermal wall losses (sets horn/port Q)
  memory_gb: null               # default 60 % of free RAM; the mesh coarsens with a warning above it
  cache_dir: .spkmod_cache

frequencies: {min: 25, max: 3000, per_octave: 24}

drivers:                         # either T/S ...
  woofer: {Fs: 35, Qes: 0.38, Qms: 5.0, Vas_l: 120, Re: 5.6, Le_mH: 1.2, Sd_cm2: 522, Xmax_mm: 7, source: "datasheet"}
  # ... or fundamental parameters (preferred; used as given when Bl is present)
  12-280-W: {Re: 6.6, Bl: 20.68, Mms_g: 85.7, Cms_mm_per_N: 0.204, Rms: 2.50, Sd_cm2: 510.7, Le_mH: 1.0}
  # optional: Mmd_g (moving mass WITHOUT air load, see section 7), L2_mH + R2 (LR-2 voice coil), Xmax_mm, notes
  pr: {passive: true, Mmd_g: 120, Cms_mm_per_N: 0.3, Rms: 1.0}     # passive radiator

cabinets:
  deep30:
    step: cad/deep30.step        # solids only (panels, braces, ports, waveguide body)
    pistons:                     # one per diaphragm, in the CAD's own frame (metres)
      - {name: drv, center: [0.508, 0.275, 0.489], axis: [0.063, 0, 0.998], sd_cm2: 510.7,
         cutout: 0.282, thickness: 0.008, motor: [0.156, 0.07]}       # motor: [diameter, depth]
      # cone dish: recess: 0.030 sets the piston back 30 mm, with the dish air open to the front and a rigid basket
      # ring (basket_wall, default 0.005) sealing it from behind; or cone_depth: 0.09 (volume Sd*depth/3 -> recess depth/3)
    stuffing: [{box: [x0, y0, z0, x1, y1, z1], sigma: 10000}]          # fibre fill, flow resistivity Pa s/m^2
    envelope: null               # default: bounding box of the solids (the FEM/BEM interface)

scenes:                          # boxes that interact acoustically are solved together
  stack:
    ground: true                 # rigid ground plane z = 0 in the BEM (boxes standing on it)
    max_freq: 500                # also min_freq
    items:
      - {cabinet: deep30, name: sub_rear_l, position: [-3.632, 0, 0], rotation: [0, 0, 0]}  # rotation about x, y, z in that order
  pole:
    ground: false                # free field; grass added at system level via image points:
    ground_reflection: [[500, 1.0], [4000, 0.5]]   # tabulated |R|(f), like the floor model ...
    # ground_sigma: 200000       # ... or Miki impedance ground (flow resistivity); omit both for rigid
    polars: true                 # far-field balloons (horizontal + vertical, 5 degree steps)
    items: [...]
  pole_hf:                       # high-frequency model of the waveguide alone (see section 7)
    exterior: baffle             # mouth radiates into an infinite baffle in its own plane: no box BEM
    ground: false
    ground_reflection: [[500, 1.0], [4000, 0.5]]
    min_freq: 3150               # first frequency above the full-box scene's max_freq: no overlap
    max_freq: 16000
    items: [{cabinet: waveguide, name: pole_wg, position: [...], rotation: [...]}]

wiring:                          # body = <item name>.<piston name>
  stack:
    sub_rear_l.drv: {driver: 12-280-W, channel: sub_rear, polarity: 1}

dsp:
  camilladsp: dsp/party_deep70_party.yml     # the config the Pi runs (biquads, crossovers, delays, gains, mixers, FIR)
  presets: {day: {camilladsp: dsp/party_deep70_day.yml}}
  outputs: {sub_front: 0, sub_rear: 1, lowmid: 2}   # channel -> CamillaDSP output index
  input: [0.5, 0.5]                          # program = weighted capture channels (mono)
  volts_at_0dbfs: 32

listeners: {file: floor_spots.csv}           # or {grid: {x: [..], y: [..], spacing: 0.25, heights: [1.4, 1.6, 1.8]}} or {points: [[x,y,z], ...]}
zones:                                        # extra named listener sets, reported separately ('floor' = listeners)
  behind: {box: [-8, -5, -3, 3], heights: [1.6], spacing: 0.5}     # box: [x0, x1, y0, y1]
  dj: {points: [[0.5, 0, 1.6]]}                                     # or {file: dj.csv}
```

## 4. Cabinet specs (panel lists → STEP)

You can write cabinets as YAML panel lists in **millimetres**, with no CAD software, then run `spkmod cad spec.yml`. That writes the STEP, a three-view preview PNG and the `pistons:` lines to paste into `project.yml`. Generators such as `examples/party_rig/make_specs.py` write these specs.

```yaml
name: deep30
units: mm
parts:                               # solids (merged)
  - {type: box, name: bottom, at: [0, 0, 0], size: [1216, 550, 18]}
  - {type: prism, name: reflector, plane: xz, points: [[1182, 18], [1182, 120], [1080, 18]], extrude: [18, 532]}
  - {type: tube, name: port, center: [x, y, z], axis: [1, 0, 0], inner_d: 100, length: 200, wall: 6}
  - {type: cylinder, name: c, center: [x, y, z], axis: [0, 0, 1], diameter: 38, length: 80}
  - {type: step, file: waveguide.step}                       # an existing STEP part
  - {type: box, ..., rotate: {about: [x, y, z], axis: [0, 1, 0], angle: 30}}
cutouts:                             # removed from all parts
  - {type: cylinder, center: [x, y, z], axis: [0, 0, 1], diameter: 282, depth: 40}
pistons:
  - {name: drv, center: [x, y, z], axis: [0, 0, 1], sd_cm2: 510.7, cutout: 282, thickness: 8, motor: [156, 70],
     recess: 30}                     # optional cone dish (mm), or cone_depth: 90; basket_wall: 5
```

Rules:
* **Panels must touch or overlap.** Any gap is an air leak.
* **Leave out anything outside the acoustic box** (handles, feet). Everything inside the bounding box that isn't solid counts as air.
* **Openings need no marking.** Mouths and ports are simply where there is no panel; they are found automatically.
* **Piston placement:** the centre goes at mid-thickness of its baffle, and `axis` is the direction the cone moves when it pushes air out of its front. Polarity depends on it.
* **Mesher help:**
  * Narrow air gaps below h/3 are listed in a warning with their position.
  * If gmsh fails, spkmod finds the piston whose frame ring or motor nearly touches a CAD face, nudges it up to ±3 mm and logs the move. Otherwise it raises an error naming the piston.
  * Check the tagging before solving with `spkmod mesh project.yml --scene stack --out stack.vtk` (ParaView: `kind` 0 = rigid, 1 = opening, 2 = vibrating).

## 5. Sweeps (variants → one table)

```yaml
name: sub_layouts                 # default: file name
project: ../project.yml
scenes: [stack]                   # default: all
frequencies: {min: 25, max: 165, per_octave: 24}    # or {list: [..]}
listeners: {file: ../floor_spots.csv}               # optional override of the project's
zones: {behind: {box: [-8, -5, -3, 3], heights: [1.6], spacing: 0.5}}
bands: {25-30: [25, 30], 30-36: [30, 36], 36-45: [36, 45], 45-60: [45, 60], 60-80: [60, 80], 80-100: [80, 100]}
                                  # or null (floor-model + sub bands), 'sixth' | 'third' | 'octave'
smoothing: 6                      # 1/n octave for spreads and holes
dsp: project                      # project (its CamillaDSP) | flat (amp volts on every channel)
reference: m2m                    # variant that d_median_db is taken against (default: first)
reference_csv: ../floor_model_deep30_x80.csv        # optional: floor zone vs the floor model
parallel: 1                       # local runs only: solves at once
variants:
  - name: m2m                                               # the project as it is
  - name: subs_only
    remove: [lowmid]
  - name: line4
    remove: [sub_rear_l, sub_rear_r]
    move: {sub_front_l: {position: [-1.216, 0, 0], rotation: [0, 0, 0]}}
    add: [{cabinet: deep30, name: extra, position: [-1.216, 0.55, 0], wiring: {drv: {driver: 12-280-W, channel: sub_front}}}]
    replace: {sub_front_r: {cabinet: fold3_35}}
  - name: gen                                               # cabinet from a generator: f(**params) -> spec dict or YAML path
    cabinets: {deep30: {generator: gen.py:sub_from_params, params: {Dx: 1.18}}}
  - name: rear_delay                                        # offline only: reuses m2m's solve
    base: m2m
    grid: {offline.delay_ms.sub_rear: [-0.5, 0, 0.5]}       # expands to rear_delay[sub_rear=-0.5] ...
    offline: {polarity: {sub_rear: -1}}                     # also gain_db, mute: [ch], preset: party, filters: {ch: [CamillaDSP filter dicts]}
  - name: finer
    project: {solver: {h_max: 0.08}}                        # any deep merge into project.yml
```

What triggers a new solve, and what doesn't:
* **New solve:** variants that change geometry, solver settings, frequencies or listeners.
* **No new solve:** offline changes (delay, polarity, gain, mute, filters, preset) and wiring or driver changes. Solves are cached by content, so a rerun only solves what is missing.

## 6. Outputs and metric definitions

| File | Content |
|---|---|
| `table.txt`, `table.csv` | Per variant × zone × band (columns below) |
| `perfreq.csv` | Zone median level and tonal p10/p90 at every analysis frequency |
| `reference.csv` | With `reference_csv`: floor vs floor model per band (offset, Δmedian, spot rms, model/ref p10, p90 and holes) |
| `fields/*.npz` | Per-channel pressure per volt at every point, excursion and impedance: offline changes need no solve |
| `variants/*.npz` | Local runs only: total pressure per variant (`spkmod.sweep.load_variant`) |

Columns of `table.csv`, per zone and band:
* `median_db`: median over spots of the band level (power average over the band), in dB SPL at 0 dBFS program through the DSP and amps. `d_median_db` is the same against the reference variant.
* `p10`, `p90`, `rms` (tonal spread): each spot is first normalised to its own overall level over the analysed bands. Its 1/6-octave-smoothed deviation from the zone median is then pooled over the band's frequencies and spots.
* `holes6_pct`, `holes3_pct`: share of spots where, inside the band, the spot's smoothed tonal deviation from the zone median dips more than 6 or 3 dB below that spot's own mean deviation over the band. Slopes shared by every spot (a high-pass, a crossover) cancel, so only local dips count.
* `x_peak_mm`: largest peak cone excursion in the band at 0 dBFS (sine), over all drivers.

Comparisons:
* `spkmod compare runs/<name> --a A --b B` gives a B − A table.
* `--variant V --reference CSV` compares a variant with a floor-model table. The reference is aligned by its median offset, because the floor model's level is arbitrary.

## 7. Conventions and known limits

* **Driver moving mass (important).** The 3D solve supplies the air load on both sides of the cone. spkmod therefore uses Mmd = Mms − 2·(8/3)ρa³ (Small's infinite-baffle air load) unless `Mmd_g` is given.
  * For the 12-280-W that removes 13.3 g (85.7 g → 72.4 g), which raises its resonances by about 9 % compared with using Mms as the moving mass.
  * A datasheet Mms measured in free air carries less air load than that, so the true Mmd is probably near 78 g.
  * Until the added-mass measurement, bracket the effect: run once with the default and once with `Mmd_g: 85.7`.
* **Phasors and levels.** All phasors are rms. Levels are dB SPL re 20 µPa. Excursion peaks are √2·|x|.
* **Ground.** On-ground scenes use a rigid ground in the BEM (correct for grass below about 300 Hz). Elevated scenes use image points with `ground_reflection` or `ground_sigma`.
* **Near-surface points.** Points closer than about one element size (h) to a box surface are less accurate. Keep zones at least h away.
* **Rigid walls, small signal.** No panel vibration, no Bl(x)/Cms(x), no thermal compression, no crowd absorption.
* **Frequency range of the 3D model.** The full-box BEM is good up to where its mesh fits in memory: 16 GB covers about 9.6k BEM nodes, or about 17k when only a few frequencies are asked per band. For the pole box that is roughly 4–6 kHz.
* **The high-frequency waveguide model (`exterior: baffle`).** For the band above that limit, give the waveguide its own cabinet:
  * The solids are the bore and its face. The only opening is the mouth, which must lie in one plane: use a STEP in a frame where the face is axis-aligned, or set `envelope:` to a box aligned with the face.
  * Wire the compression driver's piston to the same channel as in the full scene.
  * Split the frequency range so the scenes don't overlap. The full scene stops at `max_freq` (e.g. 3000); the baffled scene starts at the next analysis frequency.
  * It is valid when the face around the mouth spans several wavelengths: for the 330 × 260 mm face, from about 2–3 kHz.
  * Validated against the exact baffled-piston impedance (within 2 %, ka 0.5–3) and directivity (`tests/test_baffled.py`).
* **Shards and band starts.** In sharded runs, scenes whose `min_freq` lies above the sweep's first frequency get band boundaries from the shard. This still meets the λ/6 meshing rule but is not bit-identical to a single run.

## 8. Local install and commands

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[fast,test]"
# Linux without a display may need: apt install libglu1-mesa libxcursor1 libxinerama1 libxft2
pytest -q                                              # validation suite (~4 min)
spkmod cad cad/specs/*.yml --out cad                   # panel lists -> STEP + previews
spkmod mesh project.yml --scene stack --out stack.vtk  # check tagging / gaps
spkmod sweep sweeps/x.yml --dry-run                    # variants and solves needed
spkmod sweep sweeps/x.yml [--parallel 2] [--out runs/x]   # solve + tables locally
spkmod sweep sweeps/x.yml --out D --shards | --shard i | --merge   # batch/Slurm building blocks
spkmod compare runs/x --a A --b B
spkmod report project.yml [--reference ref.csv] [--results runs/r/results]   # impedance, excursion, polars, zones
spkmod job plan|solve|merge runs/r.yml                 # whole-project batch runs (GitHub: push runs/r.yml)
```

The fast OpenCL backend is used when it works on the machine's CPU. spkmod tests it once per CPU model in a subprocess and falls back to numba otherwise. To force a backend, set `SPKMOD_BEM_DEVICE=numba` or `opencl`.
