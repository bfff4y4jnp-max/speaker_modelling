# Brief: tooling for faster cabinet and layout work (2026-10-15)

From the design project (Peter's party rig). These are requests for whoever codes `spkmod`; the design side runs solves but does not change `spkmod/`. Work in this order and merge each item on its own, so the early ones are usable while later ones are in progress. Background on the rig and earlier findings: `examples/party_rig/README.md` and `examples/party_rig/NOTES_FOR_REPO.md`.

## How spkmod is used today (what hurts)

Every comparison is a hand-written script that builds `Cabinet` objects, calls `solve_scene`, wires with `Wiring`/`solve_network`, applies delays and polarity offline, then computes floor medians and band tables by hand. Examples the design side wrote this week:
- subs-only stack vs the 1D model (33 freqs, 25-165 Hz, 1005 listener points);
- per-row split (front row, rear row, rear row with the front boxes removed);
- level behind the stack with a rear-row delay sweep (delay applied offline to the saved per-channel fields);
- sub layouts: 2+2 mouth-to-mouth vs 4 in a line vs a 2x2 block, with floor / DJ-side / behind zones;
- subs + low-mid crossover at 80 vs 100 Hz with HP/LR4 filters and a low-mid delay search.

Timings on 4 cores at h 0.15: one deep30 horn ~100 s for 33 freqs; the 4-horn stack ~330 s. h 0.05 gave the same answer as h 0.15 for the subs (903 s vs 100 s).

## 1. Sweep and batch runner (highest priority)

A way to describe many variants of a scene and get one comparison table back, without a script per study.
- Input: a base project + scene and a list of variants. A variant can change item positions/rotations, add/remove items, swap a cabinet, or change cabinet parameters through a generator function (see `examples/party_rig/make_specs.py`, where each spec is built from a few numbers). Also offline-only variants (channel delay, polarity, gain, DSP preset) that reuse a solve.
- Solves run queued, one at a time or N in parallel within a memory budget, each cached by its existing scene hash, resumable after a crash or container restart.
- Output: one table (CSV + text) per sweep with the metrics of item 3 per variant and band, plus the saved per-channel fields (`.npz`) so further offline changes need no re-solve.
- CLI, for example `spkmod sweep sweep.yml`, and a Python API.

## 1b. Run sweeps on GitHub Actions for free, sharded across jobs (do with or right after item 1)

Peter wants to push a sweep file and get results back without anyone babysitting a machine. The repo is public, so standard GitHub-hosted runners are free with no minute limit (ubuntu-latest = 4 vCPU, 16 GB RAM). The Free plan runs up to 20 jobs at once, which is about 80 cores at no cost.
- A workflow (`workflow_dispatch` plus push of `sweeps/*.yml`) with a matrix that shards the sweep. Each job takes a chunk of frequencies and/or variants (e.g. `--shard i/N`) and solves only that chunk. A final job merges the shards' `.npz` into the result table and commits it to a `results/` branch, or uploads it as an artifact.
- Keep the per-job setup short: cache pip/venv and the numba cache, and give each shard enough work that setup (~2-3 min) isn't most of the job. Jobs time out at 6 h.
- Shard results must be mergeable: the same mesh and settings in every shard, with frequencies split cleanly, and the cache key per frequency chunk.
- Make `runs-on` a repo variable. Larger paid runners (Team plan org: $0.082/min for 32 cores, $0.162 for 64) remain an option, and then the same sharding runs as parallel worker processes inside one job.
- Print elapsed minutes per job.
## 2. Reuse solves when only placement changes

Moving a box today re-meshes and re-solves the whole scene, interior FEM included. The inside of a box is the same wherever it sits. Suggested approach (your call): mesh each cabinet once in its own frame, reduce its interior FEM to an envelope-surface operator per frequency (Schur complement / impedance on the envelope nodes), cache that per cabinet + h + frequency, and on each layout only transform it and solve the exterior BEM. Combined with item 1 this should make layout sweeps several times faster. A check against a full solve of the current stack scene should agree to ~0.1 dB.

## 3. Standard floor metrics built in

So results line up with the design project's 1D floor model (`sysmodel.py`), without per-study code:
- per listener set or named zone (floor, DJ position, behind the stack, neighbours): median level, p10/p90 spread re median, share of spots with a hole deeper than 6 dB, per frequency and per band (fractional-octave and user bands like 25-30, 30-36, 36-45, 45-60, 60-80, 80-100 Hz);
- zones defined in project.yml as boxes or point files, e.g. `zones: {dj: {box: [x0,x1,y0,y1], heights: [1.4,1.6,1.8], spacing: 0.5}}`;
- difference tables between two variants or between a solve and a reference CSV (e.g. `floor_model_deep30_x80.csv`);
- `floor_stats` already exists; extend rather than duplicate.

## 4. Mesher robustness

From NOTES_FOR_REPO.md:
- refine locally where facing walls are close (25 mm sub fold gap, 8 mm low-mid front chamber, 40 mm slot), or warn when a gap is under ~h/3;
- a piston at the throat of a curved (lofted) bore fails with gmsh "PLC Error: A segment and a facet intersect"; a 1 mm step between a motor face and a solid face fails with "1D mesh not a closed loop". Fix, or at least name the piston/part in the error;
- thin curved shells (6 mm printed waveguide wall with air both sides) fail at practical h.
Needed before the pole box and waveguide can run at high frequency.

## 5. Fast solver for large meshes (FMM)

Above ~7k BEM nodes dense assembly runs out of memory/time (4.1k nodes with Burton-Miller ~150 s per frequency). Needed for the pole box and fills toward full range. Notes on building exafmm-t are in NOTES_FOR_REPO.md.

## Small fixes (fit in whenever convenient)
- Ground as a tabulated |R|(f) for scenes solved without a rigid ground: the floor model uses |R| = 1 below 500 Hz falling to 0.5 at 4 kHz.
- Allow `source:` / `notes:` keys on drivers (today `Driver(**d)` raises TypeError).

## Not wanted
A general refactor or style pass. The solver's accuracy has checked out; speed and convenience for many variants matter most.
