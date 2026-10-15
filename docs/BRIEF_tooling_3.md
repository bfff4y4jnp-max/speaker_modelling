# Brief 3: GitHub Actions robustness (from the design project, 2026-10-15 evening)

Found while running the party-rig sweeps on free runners. Preference order: 1, 2, 3, 4, 5.

## 1. A failed shard should not lose the whole sweep
Today one failed `solve (i)` job makes `merge` skip, so no variant gets results, even when 18 of 20 shards succeeded
(runs 38808595920 and 38808653421: two low-mid variants failed meshing and the other eleven were lost).
Wanted: `merge` runs with `if: always()`, writes fields for every solve group whose shards all succeeded, writes the
tables for those variants, lists the missing variants (and why: the failing shard and its last error line) in
table.txt and the step summary, and the run still ends red. A later push or rerun then only solves what is missing
(the existing "a rerun only solves what is missing" rule).

## 2. Every changed sweep file in a push should run
`plan` solves only the first changed `**/sweeps/*.yml` and warns about the rest. Wanted: one run per changed file
(a matrix over requests, or `plan` emitting shards for all of them with per-request merge jobs). Results for each go
to runs/<name>/ as now. Concurrent result commits to the same branch must rebase and retry.

## 3. Shards that run out of runner memory
pole_lineC (pole box, 200-3000 Hz, ~5.4k BEM nodes at 3 kHz) killed the runner on its top band twice ("The runner
has received a shutdown signal" ~80 s in, no Python error): a 16 GB runner OOM. Wanted: `plan` estimates peak memory
per shard (BEM nodes^2 x 16 bytes x the assembler's working copies, plus FEM) and, above ~12 GB, either splits the
band further, drops to a cheaper BEM (FMM if item 5 of brief 1 is in) or fails at plan time with a clear message
naming the frequency where it stops fitting. Also log peak RSS per shard in the step summary.

## 4. Mesher: frame cutout equal to the hole bore
A piston whose `cutout` equals the CAD hole diameter (282 mm in a 282 mm bore) gives coincident cylinders: gmsh
"PLC Error: A segment and a facet intersect" on some geometries (low-mid v02_cav61, v12_plug), and with the native
recess on lowmid.step it meshed but added 4 aperture triangles (236 vs 232), i.e. a sliver opening at the frame.
Wanted: detect cutout within ~1 mm of a coaxial CAD bore and enlarge it to bore + 8 mm (logged), and warn when
aperture area changes when only a piston option changed.

## 5. Field keys: local dry-run vs runner
For lowmid_cone (static STEPs, before #18) the merged field files on the branch had keys that a local
`--dry-run` of the same commit did not reproduce, so the local tools could not find them by key. If #18 did not
already make keys platform-independent (gmsh/OCC version, float formatting), please do, and add a test that the
key of a fixed example matches a stored value.

## 6. Silent empty interior (0 tets) gives all-zero fields
Root cause found: in the low-mid variants with `cutout: 0.282` in a 282 mm CAD hole (cav37-based boxes),
`build_scene_mesh` returns **0 air tets** at every h (0.04-0.15) with no error. The interior operator logs
"0 tets", the pistons are unloaded (x per V ~3x too big, identical Z for every geometry) and every pressure is
exactly 0; the sweep goes green and the table shows -206 dB (runs 38811745008, 38815909571). Cutout 0.29 gives
25-48k tets. Wanted: raise an error when a cabinet with pistons meshes to 0 tets (or a piston face touches no air
on either side), name the piston, and try the item 4 auto-fix (enlarge the cutout) first. Merge should also refuse
a field whose pressure is all zero.
