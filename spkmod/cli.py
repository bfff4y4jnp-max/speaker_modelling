"""Command line: spkmod {mesh,solve,report} project.yml"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np


def _setup(verbose):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(asctime)s %(message)s")
    logging.getLogger("spkmod").setLevel(logging.INFO)
    logging.getLogger("skfem").setLevel(logging.WARNING)
    logging.getLogger("pyopencl").setLevel(logging.WARNING)
    import bempp_cl.api as b
    b.set_logging_level("warning")


def cmd_mesh(pr, a):
    from .geometry import build_scene_mesh
    sc = pr.cfg["scenes"][a.scene]
    h = min(pr.settings.h_max, pr.air.c / (a.freq * pr.settings.elements_per_wavelength))
    m = build_scene_mesh(pr.cabinets(a.scene), h, pr.settings.h_min, ground=sc.get("ground", True))
    print(m.summary())
    for n in m.body_names:
        print("  body", n)
    if a.out:
        import meshio
        cells = [("triangle", m.bem_tris)]
        meshio.write(a.out, meshio.Mesh(m.nodes, cells, cell_data={"kind": [m.bem_kind]}))
        if len(m.tets):
            base, ext = os.path.splitext(a.out)
            meshio.write(base + "_air" + ext, meshio.Mesh(m.nodes, [("tetra", m.tets)], cell_data={"region": [m.tet_region]}))
        print("wrote", a.out)


def _scenes(pr, a):
    names = [a.scene] if a.scene else list(pr.cfg["scenes"])
    return {n: pr.solve(n, a.fmin, a.fmax, a.per_octave) for n in names}


def cmd_solve(pr, a):
    for n, r in _scenes(pr, a).items():
        print(f"{n}: {len(r.freqs)} freqs, {len(r.body_names)} bodies, {r.meta.get('time_s', 0):.0f} s, {r.meta.get('meshes')}")


def cmd_report(pr, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .system import floor_stats, spl
    os.makedirs(a.out, exist_ok=True)
    res = _scenes(pr, a)
    sysm = pr.system(res)
    f = sysm.freqs
    ch = sysm.channel_responses()
    tf = pr.channel_tf(f, a.preset)
    lines = [f"# spkmod report: {os.path.basename(pr.path)}", ""]
    # per channel: impedance and excursion per volt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, d in ch.items():
        ok = np.isfinite(d["Z"]) & d["covered"]
        ax[0].semilogx(f[ok], np.abs(d["Z"][ok]), label=c)
        for b, x in d["x"].items():
            ax[1].semilogx(f[ok], 1e3 * np.sqrt(2) * np.abs(x[ok] * tf.get(c, 1)[ok] if c in tf else x[ok]), label=f"{c}:{b}")
    ax[0].set(xlabel="Hz", ylabel="|Z| ohm", title="Channel impedance")
    ax[1].set(xlabel="Hz", ylabel="peak mm", title="Excursion at 0 dBFS sine (incl. DSP)")
    for x in ax:
        x.grid(True, which="both", alpha=.3); x.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "drivers.png"), dpi=120); plt.close(fig)
    # polars per scene
    for n, r in res.items():
        if r.T_far is None:
            continue
        net = pr.wiring(n)
        from .drivers import solve_network
        nt = solve_network(r, net, pr.air)
        far = sum(d["far"] * (tf[c][np.searchsorted(f, r.freqs.round(6))][:, None] if c in tf else 1) for c, d in nt.items())
        nd = far.shape[1] // 2
        L = spl(far)                       # at 1 m (far-field extrapolated)
        ang = np.arange(-180, 180, 360 / nd)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for j, (title, sl) in enumerate((("horizontal", slice(0, nd)), ("vertical", slice(nd, 2 * nd)))):
            im = ax[j].pcolormesh(r.freqs, ang, (L[:, sl] - L[:, sl][:, [nd // 2]]).T, vmin=-30, vmax=3, shading="auto", cmap="viridis")
            ax[j].set_xscale("log"); ax[j].set(title=f"{n} {title} (re 0 deg)", xlabel="Hz", ylabel="deg")
        fig.colorbar(im, ax=ax); fig.savefig(os.path.join(a.out, f"polar_{n}.png"), dpi=120); plt.close(fig)
    # floor
    if sysm.points is not None:
        p = sysm.response(tf, ch)
        from .metrics import compare_to_reference, parse_bands, zone_table
        pts_all, zones = pr.observation_points()
        if pts_all is None or len(pts_all) != p.shape[1]:
            zones = {"floor": np.arange(p.shape[1])}       # results solved before zones were added
        p_all, pts_full = p, sysm.points
        bands = parse_bands(None, f)
        zrows = zone_table(f, p_all, zones, bands)
        p = p_all[:, zones["floor"]]
        st = floor_stats(f, p)
        fig, ax = plt.subplots(figsize=(8, 4))
        Lp = spl(p)
        ax.semilogx(f, np.median(Lp, 1), "k", label="median")
        ax.fill_between(f, np.percentile(Lp, 10, 1), np.percentile(Lp, 90, 1), alpha=.3, label="p10-p90")
        ax.set(xlabel="Hz", ylabel="dB SPL at 0 dBFS", title="Floor response (absolute)"); ax.grid(True, which="both", alpha=.3)
        ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "floor.png"), dpi=120); plt.close(fig)
        lines += ["## Zones (levels dB SPL at 0 dBFS; spreads re zone median)", "",
                  "| Zone | Band | Median | p10 | p90 | rms | holes >6 dB | >3 dB |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['zone']} | {r['band']} | {r['median_db']:.1f} | {r['p10']:+.1f} | {r['p90']:+.1f} | "
                  f"{r['rms']:.2f} | {r['holes6_pct']:.1f}% | {r['holes3_pct']:.1f}% |" for r in zrows]
        if getattr(a, "reference", None):
            cmp_ = compare_to_reference(f, p, pts_full[zones["floor"]], a.reference, bands)
            lines += ["", f"## Floor vs reference `{os.path.basename(a.reference)}` (aligned by {cmp_['offset_db']:+.1f} dB)", "",
                      "| Band | dMedian | spot rms | model p10..p90 | ref p10..p90 | holes >6 dB model/ref |",
                      "|---|---|---|---|---|---|"]
            lines += [f"| {r['band']} | {r['d_median_db']:+.1f} | {r['spot_rms_db']:.2f} | {r['model_p10']:+.1f}..{r['model_p90']:+.1f} "
                      f"| {r['ref_p10']:+.1f}..{r['ref_p90']:+.1f} | {r['model_holes6_pct']:.1f}/{r['ref_holes6_pct']:.1f}% |"
                      for r in cmp_["rows"]]
        pts = pts_full[zones["floor"]]
        z = np.unique(pts[:, 2])
        zc = z[np.argmin(np.abs(z - 1.6))]
        sel = np.isclose(pts[:, 2], zc)
        m = (f >= 50) & (f <= 8000)
        lvl = 10 * np.log10(np.mean(np.abs(p[m][:, sel]) ** 2, 0) / 4e-10)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc_ = ax.scatter(pts[sel, 0], pts[sel, 1], c=lvl - np.median(lvl), cmap="RdBu_r", vmin=-6, vmax=6, s=60, marker="s")
        ax.set_aspect("equal"); fig.colorbar(sc_, label="dB re median"); ax.set_title(f"Level 50 Hz-8 kHz at z={zc} m")
        fig.savefig(os.path.join(a.out, "coverage.png"), dpi=120); plt.close(fig)
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("report written to", a.out)


def cmd_compare(a):
    import csv
    from .metrics import compare_to_reference, diff_rows, parse_bands
    from .sweep import _safe, _write_csv, format_reference, load_variant
    if a.a and a.b:
        with open(os.path.join(a.sweep_dir, "table.csv")) as fh:
            rows = list(csv.DictReader(fh))
        ra = [r for r in rows if r["variant"] == a.a]
        rb = [r for r in rows if r["variant"] == a.b]
        if not ra or not rb:
            raise SystemExit(f"variant not in table.csv: {a.a if not ra else a.b}")
        d = diff_rows(ra, rb)
        print(f"{a.b} minus {a.a}")
        print(f"{'zone':10s} {'band':>9s} {'dMed':>6s} {'dp10':>6s} {'dp90':>6s} {'drms':>6s} {'d>6dB':>7s} {'d>3dB':>7s}")
        for r in d:
            print(f"{r['zone']:10s} {r['band']:>9s} {r['d_median_db']:+6.1f} {r['d_p10']:+6.1f} {r['d_p90']:+6.1f} "
                  f"{r['d_rms']:+6.2f} {r['d_holes6_pct']:+6.1f}% {r['d_holes3_pct']:+6.1f}%")
        if a.out:
            _write_csv(a.out, d, list(d[0]) if d else [])
    elif a.variant and a.reference:
        v = load_variant(os.path.join(a.sweep_dir, "variants", f"{_safe(a.variant)}.npz"))
        idx = v["zones"][a.zone]
        f = v["freqs"]
        res = compare_to_reference(f, v["p"][:, idx], v["points"][idx], a.reference, parse_bands(None, f))
        rows = [{"variant": a.variant, "offset_db": res["offset_db"], **r} for r in res["rows"]]
        print(format_reference(rows, os.path.basename(a.reference)))
        if a.out:
            _write_csv(a.out, rows, list(rows[0]) if rows else [])
    else:
        raise SystemExit("give --a and --b, or --variant and --reference")




def main(argv=None):
    ap = argparse.ArgumentParser(prog="spkmod")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("cad", help="panel-list YAML (mm) -> STEP + preview PNG + pistons snippet")
    pc.add_argument("spec", nargs="+")
    pc.add_argument("--out", help="output directory (default: next to the spec)")
    for name in ("mesh", "solve", "report"):
        p = sub.add_parser(name)
        p.add_argument("project")
        p.add_argument("--scene")
        p.add_argument("-v", "--verbose", action="store_true")
        if name == "mesh":
            p.add_argument("--freq", type=float, default=200.0, help="mesh for this top frequency")
            p.add_argument("--out", help="write .vtk/.msh for inspection")
        else:
            p.add_argument("--fmin", type=float); p.add_argument("--fmax", type=float)
            p.add_argument("--per-octave", type=int)
        if name == "report":
            p.add_argument("--out", default="report")
            p.add_argument("--preset")
            p.add_argument("--reference", help="floor-model CSV (x,y,z + sixth-octave levels) to compare the floor with")
    ps = sub.add_parser("sweep", help="many variants of a project -> one comparison table")
    ps.add_argument("spec")
    ps.add_argument("--parallel", type=int, help="solves at once (memory and cores are split)")
    ps.add_argument("--only", help="comma-separated variant names")
    ps.add_argument("--dry-run", action="store_true", help="list variants and the solves needed")
    ps.add_argument("-v", "--verbose", action="store_true")
    pc2 = sub.add_parser("compare", help="difference tables: two sweep variants, or a variant vs a reference CSV")
    pc2.add_argument("sweep_dir", help="sweep output directory (with table.csv and variants/)")
    pc2.add_argument("--a", help="variant A (difference = B - A)")
    pc2.add_argument("--b", help="variant B")
    pc2.add_argument("--variant", help="variant to compare with --reference")
    pc2.add_argument("--reference", help="floor-model CSV")
    pc2.add_argument("--zone", default="floor")
    pc2.add_argument("--out", help="write the table as CSV")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        cmd_compare(a)
        return
    if a.cmd == "sweep":
        _setup(a.verbose)
        from .sweep import run_sweep
        run_sweep(a.spec, a.parallel, a.only.split(",") if a.only else None, a.dry_run)
        return
    if a.cmd == "cad":
        from .cadgen import build
        for spec in a.spec:
            build(spec, a.out)
        return
    _setup(a.verbose)
    from .project import Project
    pr = Project(a.project)
    {"mesh": cmd_mesh, "solve": cmd_solve, "report": cmd_report}[a.cmd](pr, a)


if __name__ == "__main__":
    main()
