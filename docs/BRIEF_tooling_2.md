# Brief 2: fixes and options found while using the sweeps (2026-10-15 evening)

From the design project. In priority order; one PR each is fine.

## 1. Bug: generated cabinets lose their results at merge (fix first)

`examples/party_rig/sweeps/lowmid_cone.yml` (first version, branch `claude/lowmid-cone`, run 38805906771) had two variants with
`cabinets: {lowmid: {generator: ../make_specs.py:lowmid_from_params, ...}}`. All 8 shards solved, but `table.txt` only
has the `flat` variant: the merge silently dropped both generated ones. Their fields are in `runs/lowmid_cone/fields/`
under keys that the merge (and a local `--dry-run`) don't find.

Cause (as far as I can tell): `_acoustic_signature` hashes the STEP file content, and gmsh writes a date stamp into the
STEP header, so every runner that regenerates the cabinet gets a different key.
- Hash the generator's spec (the dict it returns) instead of the STEP bytes, or strip the STEP header before hashing.
- The merge should fail loudly (or at least list the variants with missing fields) instead of writing a partial table.

The workaround used now: commit static STEPs and define the cabinets inside the sweep (`project: {cabinets: ...}` +
`replace:`).

## 2. Piston recess / cone cavity (native option)

Pistons are flat discs at the given centre, so the air in a cone's dish is missing from whatever is in front of it. That
matters when the cone faces a small chamber (the low-mid's 8 mm front chamber: ~3 L of the design's 7-9 L is the two
cone dishes). An option like `pistons: [{..., recess: 0.030}]` (or `cone_depth:` with volume = Sd x depth / 3) that
sets the piston back by that depth and builds the air column plus a sealing basket ring automatically would replace the
hand-built version in `make_specs.lowmid_spec(recess_mm)`. Watch out for coincident cylinders: a frame (`cutout`) equal
to a hole diameter coincident with the hole wall gave PLC errors; we used frame = hole + 8 mm inside a 310 mm ring.

## 3. `monte_carlo:` sweep key

Driver T/S spread, ground sigma and similar inputs that only change the offline network, not the acoustic solve. Something
like `monte_carlo: {n: 64, seed: 1, vary: {drivers.12-280-W.Mms: {rel: 0.15}, drivers.12-280-W.Bl: {rel: 0.05}}, method: lhs}`
producing p5/p50/p95 rows in the table, reusing one acoustic solve.

## 4. Crowd absorber volume

A zone volume (box, height ~1.7 m) with a porous/equivalent-fluid model (e.g. Miki, flow resistivity as a parameter)
to approximate a packed dance floor in the interior FEM or as an impedance boundary, so "empty vs packed floor" can be
compared for the low-mid and pole box.
