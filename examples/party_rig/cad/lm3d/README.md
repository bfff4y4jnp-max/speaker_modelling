# Low-mid 3D variants (2026-10-15)

`lowmid_<tag>.yml` are spkmod cad specs in the same frame and panel scheme as `spkmod_inputs/cad/specs/lowmid.yml`. They come from `../lm3d_variants.py`, and `variants.json` lists the parameters. `v00_flat` is identical to today's `lowmid.yml`.

| tag | change (from v01 unless noted) | why | 1D expectation, SPL/V change at 250 / 280 / 315 / 350 Hz vs v01 |
|---|---|---|---|
| v00_flat | flat piston (today) | reference | front chamber ~3.5 L: +2.3 / +1.4 / +0.4 / +3.6 dB (optimistic top end) |
| v01_cav37 | piston 37 mm up a 282 mm bore (2.3 L each), front chamber ~7 L | new baseline (Hornresp-equivalent) | 0 |
| v02_cav61 | 61 mm bore (3.8 L each), ~10 L | worst-case cone + flange | -2.3 / -1.5 / -1.8 / -3.3 dB |
| v03_gap6 / v04_gap10 | gap 6 / 10 mm | build tolerance | gap 6: +0.4 to +0.8 dB, gap 10: -0.4 to -0.7 dB (largest at 350) |
| v05_slot37 / v06_slot43 | slot 37 / 43 mm | build tolerance | within ±0.3 dB |
| v07_divnone | no mouth divider | size of the ~270 Hz cross mode | 1D can't see it |
| v08_divlegA | divider along all of leg A | | 1D can't see it |
| v09_divfull | full centre divider (2 half-width horns) | cross mode gone, braces the panels | 1D can't see it |
| v10_w680 / v11_w720 | inner width 680 / 720 mm | mouth width trend (640 is the narrowest that fits two 12" frames) | +0.0 / +0.1 dB hold, slightly more ripple |
| v12_plug | 61 mm bore + fixed cone-dish filler on P3, 12 mm clear | improvement idea | about +1.5 / +0.9 / +0.6 / +2.6 dB vs v01, +3.8 / +2.4 / +2.4 / +5.9 vs v02 |

Notes:
- **Cone cavity is the biggest lever.** Front-chamber volume sets the 250-400 Hz top end (1D: 3.5 → 7 → 10 L takes 350 Hz from 98.2 → 94.6 → 91.3 dB/V). So the flat-piston 3D runs are probably 2-4 dB optimistic at 250-350 Hz.
- **Recess method.** Each recessed piston sits in a bore of the 282 mm cut-out diameter, walled by a `collar_a/b` prism on top of the baffle (300 x 300 mm, parallel to it). The bore volume stands in for the cone dish. The 37 and 61 mm depths are estimates. The 12-280-W cone drawing would pin them down.
- **Cylinder conventions.** The cut-out cylinders are written centred on `center` with total length `depth`, as in `lowmid.yml`. `v12_plug` uses `type: cylinder` in `parts:` (with `length:`). If spkmod has no cylinder part, skip v12 or say so.
- **Mesh.** The 8 mm gap and the bores want local refinement (see `spkmod_inputs/NOTES_FOR_REPO.md` item 1).
