# Handover: Outdoor Party Rig design project → `speaker_modelling` repo

**For:** the Claude design project that produced the *Outdoor Party Rig (rev 2)* page, and Peter.
**Repo:** https://github.com/bfff4y4jnp-max/speaker_modelling (branch `claude/serene-hawking-xra8if`)

You can't see the repo, so this file holds everything you need. Your job is to
turn the designs you already know into **input files**. The repo turns those
into 3D FEM/BEM results (it runs in Claude Code or on Peter's laptop). No CAD
package is needed: cabinets are written as a panel list in YAML.

---

## 1. What the repo does (one paragraph)

`spkmod` takes cabinet geometry and driver positions, then solves:

* the air inside each box with FEM, including folded horn paths, wall friction losses and stuffing;
* the radiation outside with BEM, including the rigid ground under the stack and boxes that interact (the mouth-to-mouth sub rows).

It then applies drivers (Thiele-Small), the real CamillaDSP config, amp
voltage and grass reflection to give, per listener spot:

* SPL, impedance and excursion;
* polars;
* the same floor statistics as your floor model (p10/p90 spread and % of spots with dips over 6 dB, in the same bands).

The aim is to check the floor model's assumptions (1D horn models, idealised
directivity, ground handling) against a full 3D solution.

## 2. What we need from you: six deliverables

| # | File | Content |
|---|---|---|
| 1 | `cad/specs/fold3_35.yml` | Tapped-horn sub, panel list (§3) |
| 2 | `cad/specs/lowmid.yml` | Low-mid folded horn, 2 × 12-280-W |
| 3 | `cad/specs/polebox.yml` | Pole box: baffle, 2 × 8FE200 chambers, DE250 position (waveguide: §3.5) |
| 4 | `cad/specs/fill.yml` | Stereo fill, 2 × PC105-4 sealed |
| 5 | `drivers.yml` | Driver parameters (§4) |
| 6 | `layout.yml` | Stack layout, listener spots, DSP channel map (§5), plus the CamillaDSP config and FIR files from the Pi if available |

If you also have **deep30 / deep27** horn designs, add them as extra specs
(`deep30.yml`, `deep27.yml`). The 12-driver comparison can then be re-run in 3D.

## 3. Cabinet spec format (one YAML file per cabinet)

### 3.1 Coordinates and units
* **Millimetres**, right-handed, for **the cabinet on its own**. Origin at one outside corner, normally the corner on the ground at the back-left.
* Pick the cabinet's natural orientation and state it in a comment. For example: *"lying flat as used in the stack; x along the 1216 mm length, mouth at x = 1216; z up"*.
* Placement in the rig is done separately in `layout.yml`, so don't put rig coordinates here.

### 3.2 What to include
* **Every solid that touches the air path**: outer panels, internal dividers and folds, reflectors and corner fillers, braces that sit in the path, port tubes, the driver baffle.
* **Panels must overlap or touch exactly.** A gap of even 0.5 mm becomes an air leak in the model. Overlapping panels are fine (they're merged), so where in doubt, overlap.
* **Leave driver cut-outs open** (`cutouts`) and declare each driver as a **piston** (§3.4). Don't model cones, baskets or magnets as panels.
* **Openings to the outside are just "no panel there"**: horn mouths, ports, a vent. The repo detects them automatically.

### 3.3 What to leave out
* Handles, feet, castors, Speakon plates, rebates, chamfers under ~10 mm, screws, glue blocks outside the air path.
* **Anything sticking out past the main box.** The model's FEM region is the bounding box of the parts, so a handle sticking out 30 mm would add a slab of fake "air". Put protruding items in a comment instead.
* Air volumes. Don't model the air; it is derived.

### 3.4 Format

```yaml
name: fold3_35                 # file stem; becomes fold3_35.step
units: mm
# orientation: lying flat as in the stack, x = length (mouth end at x = 1216), y = width, z = up
parts:                         # solids, merged together
  - {type: box, name: bottom, at: [0, 0, 0], size: [1216, 525, 18]}        # corner + size along x, y, z
  - {type: box, name: fold1, at: [18, 18, 180], size: [982, 489, 18],
     rotate: {about: [18, 18, 180], axis: [0, 1, 0], angle: 0}}            # optional rotation (deg, right-hand)
  - type: prism                 # any straight-cut slab with angled ends: a 2D polygon extruded
    name: reflector
    plane: xz                   # polygon given in (x, z); extruded along y
    points: [[1182, 18], [1182, 120], [1080, 18]]
    extrude: [18, 507]          # y from 18 to 507
  - {type: tube, name: port, center: [x, y, z], axis: [1, 0, 0], inner_d: 100, length: 200, wall: 6}
  - {type: cylinder, name: pole_cup, center: [x, y, z], axis: [0, 0, 1], diameter: 38, length: 80}
  - {type: step, file: waveguide.step}     # an existing STEP part, e.g. the printed waveguide
cutouts:                        # removed from all parts
  - {type: cylinder, name: drv_hole, center: [600, 262.5, 189], axis: [0, 0, 1], diameter: 282, depth: 40}
  - {type: box, name: slot, at: [x, y, z], size: [dx, dy, dz]}
pistons:                        # one per driver diaphragm
  - name: drv
    center: [600, 262.5, 189]   # centre of the cone, mid-thickness of the baffle it sits in
    axis: [0, 0, 1]             # direction the cone moves when the driver pushes air OUT of its front
    sd_cm2: 522                 # effective piston area from the datasheet
    cutout: 282                 # baffle hole diameter, mm (the frame fills the hole around the piston)
    thickness: 8                # mm, keep below the baffle thickness
    motor: [170, 100]           # magnet diameter, depth in mm (behind the cone): it displaces air in the path
```

Rules for pistons:
* The centre must sit **inside the baffle thickness**, normally at mid-thickness.
* `axis` is the forward direction (front of the cone). For tapped horns this tells the model which side of the baffle is "front"; get it right, because polarity depends on it.
* For **two drivers in one box** (low-mid, pole box, fills) give each its own piston with a distinct `name`.
* **The DE250 is a piston at the waveguide throat.** Use a diameter equal to the throat exit (25.4 mm for 1"), an axis along the waveguide axis, and a small thickness (4 mm). The compression driver's own internal response comes from its datasheet.

### 3.5 The pole-box waveguide (120° × 70°, printed)
* If you have its STEP (from the printing workflow), reference it as `{type: step, file: waveguide.step}` and give its placement in the comments.
* If not, describe it with prisms or a simplified version: flat walls at the right angles from the throat to the mouth. This is accurate for coverage below about 3 kHz.
* Say which you did. Full-range waveguide modelling (to 16 kHz) needs the exact shape.

### 3.6 A complete worked example
`examples/cad_specs/generic_tapped_horn.yml` in the repo is a generic three-pass tapped horn (not fold3_35). It converts and solves cleanly:

* the mouth and driver were detected automatically;
* the horn tuning shows as a cone-excursion null at ~76 Hz, with the mouth volume velocity 23× the cone's.

Use the same style.

```yaml
name: generic_tapped_horn
units: mm
parts:
  - {type: box, name: bottom,   at: [0, 0, 0],      size: [1200, 500, 18]}
  - {type: box, name: top,      at: [0, 0, 532],    size: [1200, 500, 18]}
  - {type: box, name: side_l,   at: [0, 0, 0],      size: [1200, 18, 550]}
  - {type: box, name: side_r,   at: [0, 482, 0],    size: [1200, 18, 550]}
  - {type: box, name: back,     at: [0, 0, 0],      size: [18, 500, 550]}
  - {type: box, name: divider1, at: [18, 18, 180],  size: [982, 464, 18]}    # driver baffle
  - {type: box, name: divider2, at: [200, 18, 360], size: [1000, 464, 18]}
  - {type: box, name: front_lo, at: [1182, 18, 18], size: [18, 464, 342]}
  - {type: prism, name: reflector, plane: xz, points: [[1182, 18], [1182, 120], [1080, 18]], extrude: [18, 482]}
cutouts:
  - {type: cylinder, name: drv_hole, center: [600, 250, 189], axis: [0, 0, 1], diameter: 282, depth: 40}
pistons:
  - {name: drv, center: [600, 250, 189], axis: [0, 0, 1], sd_cm2: 522, cutout: 282, thickness: 8, motor: [170, 100]}
```

### 3.7 Self-check before handing over (do this for each spec)
1. List the air path in words from the driver to the mouth, pass by pass, with channel cross-sections (mm × mm) and path lengths. Check it against your 1D horn model (S1…S4, L12…L34, Vtc/Vrc as in Hornresp).
2. Confirm the internal air volume: outer box volume − sum of panel volumes ≈ your design's net volume. The repo prints the wood volume, so tell Peter the expected number.
3. Confirm every internal panel spans the full inner width (or say why not). Partial-width panels leak.
4. State the mouth opening size and position, and each piston's front side.

## 4. `drivers.yml`

Give one of the two forms below per driver. Use values from the datasheet, or better, from Peter's measurements, and **say which**.

```yaml
12-280-W: {Fs: , Qes: , Qms: , Vas_l: , Re: , Le_mH: , Sd_cm2: , Xmax_mm: , source: "datasheet <url> / measured <date>"}
8FE200:   {Fs: , Qes: , Qms: , Vas_l: , Re: , Le_mH: , Sd_cm2: , Xmax_mm: , source: }
PC105-4:  {Fs: , Qes: , Qms: , Vas_l: , Re: , Le_mH: , Sd_cm2: , Xmax_mm: , source: }
# or the fundamental form (preferred when known):
DE250:    {Re: , Bl: , Mms_g: , Cms_mm_per_N: , Rms: , Sd_cm2: , Le_mH: , source: }
```

Optional, if known: `L2_mH` and `R2` (the LR-2 voice-coil model), and `Mmd_g`
(moving mass without air load). Whatever parameters your floor model used,
give exactly those, so the 3D results compare like for like.

## 5. `layout.yml`

Rig coordinates: **x forward onto the floor, y to the left, z up, ground at z = 0**. The origin is the centre of the front edge of the stack on the ground.

```yaml
stack:                         # everything standing on the ground: solved together
  - {cabinet: fold3_35, name: sub_rear_l, position: [x, y, z], rotation: [rx, ry, rz]}  # metres, degrees about x then y then z (applied about the cabinet origin, then translated)
  - ...
  - {cabinet: lowmid, name: lowmid, position: [...], rotation: [...]}
pole:  [{cabinet: polebox, name: pole, position: [...], rotation: [...]}]                # include the downward aim
fills: [{cabinet: fill, name: fill_l, position: [...], rotation: [...]}, {...}]
channels:                      # amp channel -> drivers, with polarity
  sub_front: [sub_front_l.drv, sub_front_r.drv]
  sub_rear:  {drivers: [sub_rear_l.drv, sub_rear_r.drv], polarity: -1}   # or done in DSP: say which
  lowmid: [...]
  mid_low: [pole.mid_low]
  mid: [pole.mid]
  hf: [pole.hf]
  fill_l: [...]
  fill_r: [...]
dsp_outputs: {sub_front: 0, sub_rear: 1, ...}   # CamillaDSP output index per channel
volts_at_0dbfs: 32
listener_spots: "floor_spots.csv"   # x,y,z in metres: the floor model's 335 spots x 3 heights (1005 rows)
ground: {flow_resistivity: 200000}  # Pa s/m^2 used for grass in the floor model
```

Also please give:
* **The floor model's own results for the same configuration**, as a CSV: per spot and per sixth-octave band, plus the band table from the page. These are the cross-check.
* **The CamillaDSP YAML and FIR files** as on the Pi (the 16,384-tap FIR as a raw/text/wav file). If the FIR is not finalised, give the crossover, EQ and delay settings in words.

## 6. How it gets into the repo

Either:
* Peter uploads the files on github.com: repo → **Add file → Upload files** into
  `examples/party_rig/` (specs into `examples/party_rig/cad/specs/`), on branch
  `claude/serene-hawking-xra8if`; or
* Peter pastes them into the Claude Code session that built the repo.

Claude Code then merges `drivers.yml` and `layout.yml` into `examples/party_rig/project.yml` and runs, in Claude Code or on the laptop:

```bash
spkmod cad examples/party_rig/cad/specs/*.yml --out examples/party_rig/cad   # STEP + preview PNGs + pistons snippets
spkmod mesh examples/party_rig/project.yml --scene stack --out stack.vtk     # check: mouths found, all drivers listed
spkmod solve examples/party_rig/project.yml --scene stack --per-octave 12    # quick look
spkmod report examples/party_rig/project.yml --out report                    # floor stats, coverage, polars, excursion
```

`spkmod cad` writes a three-view preview PNG per cabinet (panels in grey,
pistons and their forward arrows in orange). **Send the PNGs back to the
design project to confirm the geometry matches the design.**

## 7. What comes back to the design project

* The band table in the floor model's format (spread p10/p90 and dips >6 dB, grass), from 3D instead of 1D.
* Coverage maps, sub cardioid rejection behind the stack, the sub-to-low-mid and low-mid-to-mid handovers, and pole-box polars.
* Per-driver excursion and impedance at the calibration level, for the limiter and headroom analysis (`twoview.py` can take these).
* Where the 3D results differ from the floor model by more than about 1 dB, the cause, and a recommended change to the design or DSP.

## 8. Prompt Peter can paste into the design project

> I've built a 3D FEM/BEM modelling repo for the rig. Read the attached
> HANDOVER.md and produce the six deliverables in §2. Use the exact YAML
> formats in §3–§5, in mm for cabinet specs and metres for the layout. Do the
> self-check in §3.7 for every cabinet and include it as comments at the top
> of each spec. Use the same driver parameters, layout, grass value and floor
> spots as the floor model behind the rev 2 page, and say where any value is a
> guess. Output each file as a separate code block with its file name.
