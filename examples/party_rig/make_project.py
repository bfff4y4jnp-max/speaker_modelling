"""Write examples/party_rig/project.yml, floor_spots.csv and the floor-model cross-check CSVs for the
current build (Outdoor Party Rig rev 2 with 4 x deep30 horns, sub / low-mid crossover 80 Hz, _deep70 DSP).

Sources (Subwoofer Design project, /mnt/project-files):
  outdoor_system/system.py + sysmodel.py   layout, floor spots, grass, run flags of the build
  outdoor_system/sys_int_350_rf0.5_4ohm_bs_lmb_fold3_m2m_d30_hp34_x4.5mm_up_x180.json   floor-model result (x1=80)
  outdoor_system/outdoor_party_rev2_deep70_*.yml, fir_mono_48k_deep70.wav             CamillaDSP presets + FIR
  outdoor_sub/acoustic_net.py (12-280-W), outdoor_tops/pole_box_data.py (8FE200 4 ohm),
  outdoor_fills/fill_data.py (PC105-4)                                                 driver data

Floor model frame: x across (right), y forward, z up, origin = front face of the front sub row on the ground.
spkmod frame:      x forward, y left, z up. So x_spk = y_sys, y_spk = -x_sys, same origin.
usage: python make_project.py [/mnt/project-files]
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PF = sys.argv[1] if len(sys.argv) > 1 else "/mnt/project-files"
SYS = os.path.join(PF, "outdoor_system")
RUN = "sys_int_350_rf0.5_4ohm_bs_lmb_fold3_m2m_d30_hp34_x4.5mm_up_x180.json"


def rotmat(r):
    ax, ay, az = np.radians(r)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                                  # spkmod/geometry.py _rotmat


def v(x, nd=4):
    return "[%s]" % ", ".join(str(round(float(a), nd)) for a in x)


# ------------------------------------------------------------------ fills: aim at (0, 3, 1.6) system = (3, 0, 1.6) spkmod
fill_items = []
for name, ys in (("fill_l", 4.0), ("fill_r", -4.0)):
    src = np.array([1.2, ys, 2.6])                       # acoustic centre between the two drivers (system (-+4, 1.2, 2.6))
    aim = np.array([3.0, 0.0, 1.6]) - src
    yaw = np.degrees(np.arctan2(aim[1], aim[0]))
    pitch = np.degrees(np.arctan2(-aim[2], np.hypot(aim[0], aim[1])))   # +ve = nose down (rotation about +y)
    rot = [0.0, round(pitch, 2), round(yaw, 2)]
    c_local = np.array([0.174, 0.070, 0.280])           # fill.yml: midpoint of pistons a and b (m)
    pos = src - rotmat(rot) @ c_local
    fill_items.append((name, pos, rot))

# ------------------------------------------------------------------ floor spots (sysmodel.py grid, 335 spots x 3 ears)
R = json.load(open(os.path.join(SYS, RUN)))
px, py, pz = (np.array(R[k]) for k in ("px", "py", "pz"))
spots = np.column_stack([py, -px, pz])
np.savetxt(os.path.join(HERE, "floor_spots.csv"), spots, delimiter=",", fmt="%.3f",
           header="x_m,y_m,z_m (spkmod frame; floor model sysmodel.py: 335 spots on a half-disc r 3.57 m x ears 1.4/1.6/1.8 m)")
B = np.array(R["B"])                                     # (sixth-octave band, spot) dB after the global EQ, grass
cen = np.array(R["cen"])
with open(os.path.join(HERE, "floor_model_deep30_x80.csv"), "w") as fh:
    fh.write("# floor model (outdoor_system/system.py, integration thread) for the current build: %s\n" % R["tag"])
    fh.write("# level per spot and sixth-octave band, dB re arbitrary (same reference for every spot), all channels, grass ground\n")
    fh.write("# delays (s) %s\n" % json.dumps({k: round(x, 6) for k, x in R["d"].items()}))
    fh.write("x_m,y_m,z_m," + ",".join("%.1f" % c for c in cen) + "\n")
    for i in range(len(px)):
        fh.write("%.3f,%.3f,%.2f," % tuple(spots[i]) + ",".join("%.2f" % b for b in B[:, i]) + "\n")
shutil.copy(os.path.join(SYS, "log_d30_x180.txt"), os.path.join(HERE, "floor_model_deep30_x80_bands.txt"))

# ------------------------------------------------------------------ DSP: the three _deep70 presets + FIR (path made relative)
os.makedirs(os.path.join(HERE, "dsp"), exist_ok=True)
for p in ("day", "party", "peak"):
    s = open(os.path.join(SYS, f"outdoor_party_rev2_deep70_{p}.yml")).read()
    s = s.replace("/etc/camilladsp/fir_mono_48k_deep70.wav", "fir_mono_48k_deep70.wav")
    open(os.path.join(HERE, "dsp", f"party_deep70_{p}.yml"), "w").write(s)
shutil.copy(os.path.join(SYS, "fir_mono_48k_deep70.wav"), os.path.join(HERE, "dsp"))

# ------------------------------------------------------------------ drivers
w = 2 * np.pi * 38.035
d12 = dict(Re=6.6, Bl=20.681, Mms=0.085706, Cms=204.291e-6, Sd=0.051071)
rms12 = w * d12["Mms"] / 8.187
qes12 = w * d12["Mms"] * d12["Re"] / d12["Bl"] ** 2
vas12 = 1.204 * 343.0 ** 2 * d12["Sd"] ** 2 * d12["Cms"] * 1e3
w8 = 2 * np.pi * 80
cms8, rms8 = 1 / (w8 ** 2 * 0.0153), w8 * 0.0153 / 13.0
wp = 2 * np.pi * 81.4
cmsp, rmsp = 1 / (wp ** 2 * 0.0042), wp * 0.0042 / 2.35

Y = f"""# Outdoor Party Rig, current build: 4 x deep30 tapped horns (2+2 mouth to mouth, rear row inverted
# +3.50 ms), sub / low-mid crossover 80 Hz, straight-cut low-mid horn (2 x 12-280-W), pole box (2 x 8FE200 4 ohm
# + DE250 on a 120 x 70 waveguide), M/S fills (2 x PC105-4 each). Generated by make_project.py from the
# Subwoofer Design project (integration floor model run log_d30_x180.txt) - edit the generators, not this file.
#
# Coordinates: x forward onto the dance floor, y left, z up, ground at z = 0. Origin = the floor model's origin:
# the front face of the front sub row (its back wall, because that row faces backwards), on the centre line.
# Floor model frame (x across, y forward) maps as x_spk = y_sys, y_spk = -x_sys.

air: {{temperature: 18, humidity: 70}}

solver:
  elements_per_wavelength: 6
  h_max: 0.15
  memory_gb: null            # auto: 60 % of free RAM (16 GB laptop or cloud container)
  cache_dir: .spkmod_cache

# Quick look: per_octave 12. Final: 24 (horn resonances are sharp).
frequencies: {{min: 25, max: 3000, per_octave: 12}}

# Driver data = exactly what the floor model used (same source per driver), so 3D and 1D compare like for like.
# Fs / Qes / Qms / Vas_l are listed for reference; spkmod uses the fundamental parameters when Bl is given.
drivers:
  # the box 12-280/8-W: Thomann LMS data as used by the sub and low-mid threads (outdoor_sub/acoustic_net.py D12).
  # Not yet measured: Peter's T/S + added-mass test of the first driver should replace it.
  12-280-W: {{Re: 6.6, Bl: 20.681, Mms_g: 85.706, Cms_mm_per_N: {d12['Cms'] * 1e3:.6f}, Rms: {rms12:.4f}, Sd_cm2: 510.71, Le_mH: 1.0,
             Xmax_mm: 4.0, Fs: 38.035, Qes: {qes12:.3f}, Qms: 8.187, Vas_l: {vas12:.1f}}}
  # FaitalPRO 8FE200 4-ohm version as modelled by the tops thread (outdoor_tops/pole_box_data.py '8FE200-4'):
  # Fs 80, Re 3.0, Qes 0.47, Qms 13, Sd 191 cm2, Mms 15.3 g, Bl 7.0, Le 0.25 mH (datasheet-derived, not measured).
  8FE200: {{Re: 3.0, Bl: 7.0, Mms_g: 15.3, Cms_mm_per_N: {cms8 * 1e3:.4f}, Rms: {rms8:.4f}, Sd_cm2: 191, Le_mH: 0.25, Xmax_mm: 4.62,
           Fs: 80, Qes: 0.47, Qms: 13.0}}
  # Dayton PC105-4 as modelled by the fills thread (outdoor_fills/fill_data.py, datasheet).
  PC105-4: {{Re: 3.3, Bl: 3.3, Mms_g: 4.2, Cms_mm_per_N: {cmsp * 1e3:.4f}, Rms: {rmsp:.4f}, Sd_cm2: 52.8, Le_mH: 0.32, Xmax_mm: 2.0,
            Fs: 81.4, Qms: 2.35, Vas_l: 3.68}}
  # DE250: PLACEHOLDER, NOT DATASHEET VALUES. The floor model never used T/S for it: it used 108.5 dB / 2.83 V / 1 m
  # on the waveguide after EQ, a ~700 Hz horn cut-off and a 120 x 70 pattern. These values only give the throat
  # piston a smooth velocity above ~1 kHz, so trust the hf coverage/polars, not its absolute level.
  DE250: {{Re: 5.6, Bl: 5.5, Mms_g: 0.35, Cms_mm_per_N: 0.113, Rms: 1.0, Sd_cm2: 5.07, Le_mH: 0.05}}

cabinets:
  deep30:                            # BUILD sub: 1216 x 550 x 715 mm tapped horn (cad/specs/deep30.yml)
    step: cad/deep30.step
    pistons:
      - {{name: drv, center: [0.5079, 0.275, 0.489], axis: [0.0629, 0.0, 0.998], sd_cm2: 510.7, cutout: 0.282, thickness: 0.008, motor: [0.156, 0.07]}}
  fold3_35:                          # previous sub (rev 10), for comparison: 1216 x 550 x 525 mm
    step: cad/fold3_35.step
    pistons:
      - {{name: drv, center: [0.5826, 0.275, 0.3473], axis: [0.0329, 0.0, 0.9995], sd_cm2: 510.7, cutout: 0.282, thickness: 0.008, motor: [0.156, 0.07]}}
  deep27:                            # alternative deep horn: 1216 x 550 x 700 mm
    step: cad/deep27.step
    pistons:
      - {{name: drv, center: [0.4084, 0.275, 0.4999], axis: [0.0665, 0.0, 0.9978], sd_cm2: 510.7, cutout: 0.282, thickness: 0.008, motor: [0.156, 0.07]}}
  lowmid:                            # 496 x 676 x 468 mm folded horn, 2 x 12-280-W, cones fire down into the front chamber
    step: cad/lowmid.step
    pistons:
      - {{name: drv_a, center: [0.2604, 0.178, 0.2819], axis: [0.2425, 0.0, -0.9701], sd_cm2: 510.7, cutout: 0.282, thickness: 0.008, motor: [0.156, 0.07]}}
      - {{name: drv_b, center: [0.2604, 0.498, 0.2819], axis: [0.2425, 0.0, -0.9701], sd_cm2: 510.7, cutout: 0.282, thickness: 0.008, motor: [0.156, 0.07]}}
  polebox:                           # 220 (397 with the wedge) x 360 x 850 mm, upright; waveguide in cad/waveguide_face.step
    step: cad/polebox.step
    pistons:
      - {{name: mid_low, center: [0.2125, 0.18, 0.135], axis: [1, 0, 0], sd_cm2: 191, cutout: 0.186, thickness: 0.008, motor: [0.12, 0.055]}}
      - {{name: mid, center: [0.2125, 0.18, 0.385], axis: [1, 0, 0], sd_cm2: 191, cutout: 0.186, thickness: 0.008, motor: [0.12, 0.055]}}
      - {{name: hf, center: [0.2246, 0.18, 0.7485], axis: [0.7314, 0.0, -0.682], diameter: 0.0254, cutout: 0.036, thickness: 0.004, motor: [0.1, 0.055]}}
    stuffing:                        # 25 mm foam/felt on the back and sides of each 8" chamber (cut list)
      - {{box: [0.015, 0.015, 0.015, 0.040, 0.345, 0.250], sigma: 10000}}
      - {{box: [0.040, 0.015, 0.015, 0.205, 0.040, 0.250], sigma: 10000}}
      - {{box: [0.040, 0.320, 0.015, 0.205, 0.345, 0.250], sigma: 10000}}
      - {{box: [0.015, 0.015, 0.265, 0.040, 0.345, 0.495], sigma: 10000}}
      - {{box: [0.040, 0.015, 0.265, 0.205, 0.040, 0.495], sigma: 10000}}
      - {{box: [0.040, 0.320, 0.265, 0.205, 0.345, 0.495], sigma: 10000}}
  fill:                              # 180 x 140 x 560 mm sealed, 2 x PC105-4 110 mm apart
    step: cad/fill.step
    pistons:
      - {{name: a, center: [0.174, 0.07, 0.225], axis: [1, 0, 0], sd_cm2: 52.8, cutout: 0.093, thickness: 0.006, motor: [0.07, 0.04]}}
      - {{name: b, center: [0.174, 0.07, 0.335], axis: [1, 0, 0], sd_cm2: 52.8, cutout: 0.093, thickness: 0.006, motor: [0.07, 0.04]}}
    stuffing:                        # loose wadding through the box (fills page); sigma is a guess
      - {{box: [0.012, 0.012, 0.012, 0.150, 0.128, 0.548], sigma: 3000}}

scenes:
  # Everything on the ground interacts strongly below ~400 Hz (mouth-to-mouth sub rows 1.2 m apart, low-mid
  # mouth 0.45 m in front of the stack face): ONE scene on rigid ground (the floor model's grass is |R| = 1
  # below 500 Hz too).
  stack:
    ground: true
    max_freq: 500
    items:
      # rear row: mouths face forward at x = -2.416 (box x -3.632..-2.416); polarity inverted + 3.50 ms in the DSP
      - {{cabinet: deep30, name: sub_rear_l,  position: [-3.632, 0.0, 0.0],   rotation: [0, 0, 0]}}
      - {{cabinet: deep30, name: sub_rear_r,  position: [-3.632, -0.55, 0.0], rotation: [0, 0, 0]}}
      # front row faces BACKWARDS: mouths at x = -1.216, back walls form the stack face at x = 0
      - {{cabinet: deep30, name: sub_front_l, position: [0.0, 0.55, 0.0],    rotation: [0, 0, 180]}}
      - {{cabinet: deep30, name: sub_front_r, position: [0.0, 0.0, 0.0],     rotation: [0, 0, 180]}}
      # low-mid mouth plane at x = 0.95 (floor model mouth (0, 0.95, 0.13)); its back is 0.454 m from the stack face
      - {{cabinet: lowmid, name: lowmid, position: [0.454, -0.338, 0.0], rotation: [0, 0, 0]}}
  # Pole box: free-field BEM, grass reflection added at system level. Upright on a pole standing in the low-mid's
  # lid socket; baffle front at x = 0.95 (above the low-mid mouth), box bottom 2.690 m, 8"s at 2.825 / 3.075 m,
  # waveguide mouth centre 3.36 m, axis 43 deg down (aimed near (2.5, 0, 1.6)). Full range: max_freq 16000
  # needs the printed waveguide STEP and solver.h_max ~0.004 for this scene (slow).
  pole:
    ground: false
    ground_sigma: 200000             # Delany-Bazley grass; the floor model used |R| 1 -> 0.5 from 500 Hz to 4 kHz
    min_freq: 200
    max_freq: 3000
    items:
      - {{cabinet: polebox, name: pole, position: [0.73, -0.18, 2.69], rotation: [0, 0, 0]}}
  # Fills at the front corners (system (+-4, 1.2, 2.6) m), aimed at (3, 0, 1.6) here.
  fills:
    ground: false
    ground_sigma: 200000
    min_freq: 250
    max_freq: 3000
    items:
"""
for name, pos, rot in fill_items:
    Y += f"      - {{cabinet: fill, name: {name}, position: {v(pos)}, rotation: {v(rot, 2)}}}\n"
Y += """
wiring:                              # body = <item name>.<piston name>; polarity of the rear row is done in the DSP
  stack:
    sub_rear_l.drv:  {driver: 12-280-W, channel: sub_rear}
    sub_rear_r.drv:  {driver: 12-280-W, channel: sub_rear}
    sub_front_l.drv: {driver: 12-280-W, channel: sub_front}
    sub_front_r.drv: {driver: 12-280-W, channel: sub_front}
    lowmid.drv_a:    {driver: 12-280-W, channel: lowmid}
    lowmid.drv_b:    {driver: 12-280-W, channel: lowmid}
  pole:
    pole.mid_low: {driver: 8FE200, channel: mid_low}
    pole.mid:     {driver: 8FE200, channel: mid}
    pole.hf:      {driver: DE250,  channel: hf}
  fills:
    fill_l.a: {driver: PC105-4, channel: fill_l}
    fill_l.b: {driver: PC105-4, channel: fill_l}
    fill_r.a: {driver: PC105-4, channel: fill_r}
    fill_r.b: {driver: PC105-4, channel: fill_r}

dsp:
  # Integration thread's recommended _deep70 presets (70 ms FIR); party = +3 dB kick peak, HP 38 Hz. Model values,
  # not yet on a Pi. Fills are not in this config yet (outputs 6/7 silent spares): the fills scene gets no
  # signal until the integration thread adds their channels (S = (L-R)/2, HP 300 Hz, horn + 14.5 ms, -6 dB).
  camilladsp: dsp/party_deep70_party.yml
  presets:
    day:  {camilladsp: dsp/party_deep70_day.yml}
    peak: {camilladsp: dsp/party_deep70_peak.yml}
  outputs: {sub_front: 0, sub_rear: 1, lowmid: 2, mid_low: 3, mid: 4, hf: 5, fill_l: 6, fill_r: 7}
  input: [0.5, 0.5]                  # mono program; use [1, 0] / [0, 1] to study the stereo fills
  volts_at_0dbfs: 32                 # bridged TPA3255 at 48 V, ~32 V rms

listeners:                           # the floor model's 335 spots x 3 ear heights (1005 rows)
  file: floor_spots.csv
"""
open(os.path.join(HERE, "project.yml"), "w").write(Y)
print("fills:", [(n, v(p), r) for n, p, r in fill_items])
print("spots", spots.shape, "x", spots[:, 0].min(), spots[:, 0].max(), "y", spots[:, 1].min(), spots[:, 1].max())
