"""
Every experiment and figure in the paper.

    python3 experiments.py verify    fast renderers match brute force
    python3 experiments.py control   shape fidelity / coverage / convexity studies
    python3 experiments.py bench     runtime vs KD-tree Voronoi
    python3 experiments.py figures   all paper figures
    python3 experiments.py all       everything (~3 min)

Results -> results/*.json, figures -> figures/*.png
"""
from __future__ import annotations

import argparse, gc, json, platform, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import moroni as M

HERE = Path(__file__).resolve().parent
FIG = ROOT / "paper" / "figures"
RES = HERE / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = [(640, 384), (900, 560), (1280, 720), (1600, 900)]
ROUNDS, WARMUP = 10, 3
DPI = 150
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "figure.titlesize": 18})


def _save(fig, name):
    fig.savefig(FIG / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"  figures/{name}")


def _dump(o, name):
    (RES / name).write_text(json.dumps(o, indent=1)); print(f"  results/{name}")


def _show(ax, lbl, title=None):
    img = M.colorize(np.maximum(lbl, 0))
    img[lbl < 0] = 0.07
    ax.imshow(img); ax.axis("off")
    if title: ax.set_title(title, fontsize=15)


def _render(field, xv, yv, margin=0.0, **kw):
    return M.apply_margin(M.BinnedRenderer(field, **{k: v for k, v in kw.items()
                          if k in ("alpha", "beta", "gamma", "lam")}).render(xv, yv, **kw),
                          field, xv, yv, margin)


# ------------------------------------------------------------------- verify
def cmd_verify():
    print("== verify ==")
    f = M.make_cells(); out = {}
    for w, h in RESOLUTIONS:
        xv, yv = M.grid(w, h, f.bbox)
        ref = M.render_reference(f, xv, yv)
        pr, ps = M.render_pruned(f, xv, yv, return_stats=True)
        br = M.BinnedRenderer(f); bl, bs = br.render(xv, yv, return_stats=True)
        rec = {"pruned_exact": bool(np.array_equal(ref, pr)),
               "pruned_delta": ps["delta"], "live_cells": ps["mean_candidates"],
               "ring": bs["ring"], "agreement_pct": float(100 * np.mean(ref == bl)),
               "pixels_per_ring": [n for _, n, _ in bs["pixels_per_ring"]]}
        out[f"{w}x{h}"] = rec
        print(f"  {w}x{h}: pruned exact={rec['pruned_exact']} (Δ={rec['pruned_delta']:.2f}, "
              f"{rec['live_cells']:.1f}/{f.n} live)  binned={rec['agreement_pct']:.4f}% "
              f"ring={rec['ring']} escalation={rec['pixels_per_ring']}")
    _dump(out, "verify.json"); return out


# ------------------------------------------------------------------ control
def cmd_control():
    print("== control studies ==")
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    out = {}

    # --- shape fidelity: does vertex count actually control cell shape?
    print("  shape fidelity (IoU of cell with its own polygon)")
    rows = []
    for n in (3, 4, 5, 6, 8):
        f = M.make_cells(sides=(n, n + 1), radius=0.62, jitter=0.05, seed=7)
        xv, yv = M.grid(760, 480, f.bbox)
        raw = M.render_reference(f, xv, yv, gamma=0.0)
        rec = {"n": n,
               "voronoi": M.shape_fidelity(M.voronoi_kdtree(f, xv, yv), f, xv, yv),
               "partition": M.shape_fidelity(M.apply_margin(raw, f, xv, yv, None), f, xv, yv),
               "margin0": M.shape_fidelity(M.apply_margin(raw, f, xv, yv, 0.0), f, xv, yv)}
        rows.append(rec)
        print(f"    n={n}: voronoi={rec['voronoi']:.3f}  partition={rec['partition']:.3f}  "
              f"margin=0 -> {rec['margin0']:.3f}")
    out["shape_fidelity"] = rows

    # --- coverage vs margin (mortar width)
    print("  coverage vs margin")
    f = M.make_cells(radius=0.55, jitter=0.2, seed=7)
    xv, yv = M.grid(760, 480, f.bbox)
    raw = M.render_reference(f, xv, yv)
    cov = []
    for m in (-0.10, -0.05, 0.0, 0.05, 0.12, 0.25):
        l = M.apply_margin(raw, f, xv, yv, m)
        cov.append({"margin": m, "coverage": float(np.mean(l >= 0))})
    out["coverage"] = cov
    print("    " + "  ".join(f"m={c['margin']:+.2f}:{100*c['coverage']:.0f}%" for c in cov))

    # --- convexity / connectivity vs gamma
    print("  reachable-set test vs gamma")
    f = M.make_cells(); xv, yv = M.grid(900, 560, f.bbox)

    def geom(lbl):
        comps, sol = [], []
        for i in range(int(lbl.max()) + 1):
            m = lbl == i
            if m.sum() < 50: continue
            comps.append(ndimage.label(m)[1])
            ys, xs = np.nonzero(m)
            try: sol.append(m.sum() / ConvexHull(np.column_stack([xs, ys]).astype(float)).volume)
            except Exception: pass
        c, s = np.array(comps), np.array(sol)
        return {"disconnected_pct": float(100 * np.mean(c > 1)),
                "mean_components": float(c.mean()),
                "median_solidity": float(np.median(s)),
                "nonconvex_pct": float(100 * np.mean(s < 0.98))}

    gammas = [0.0, 0.05, 0.10, 0.20, 0.35]
    out["reachable"] = {"voronoi": geom(M.voronoi_kdtree(f, xv, yv)), "moroni": {}}
    print(f"    {'':16s} {'solidity':>9s} {'nonconvex':>10s} {'disconn':>8s}")
    v = out["reachable"]["voronoi"]
    print(f"    {'Voronoi':16s} {v['median_solidity']:9.3f} {v['nonconvex_pct']:9.1f}% {v['disconnected_pct']:7.1f}%")
    for g in gammas:
        r = geom(M.render_reference(f, xv, yv, gamma=g))
        out["reachable"]["moroni"][f"{g:.2f}"] = r
        print(f"    {'gamma=%.2f' % g:16s} {r['median_solidity']:9.3f} "
              f"{r['nonconvex_pct']:9.1f}% {r['disconnected_pct']:7.1f}%")
    _dump(out, "control.json")

    # --- figures for the above
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ns = [r["n"] for r in rows]
    ax[0].plot(ns, [r["margin0"] for r in rows], "o-", lw=2.5, ms=9, c="#27ae60", label="MORONI (margin=0)")
    ax[0].plot(ns, [r["partition"] for r in rows], "s--", lw=2, ms=8, c="#e67e22", label="MORONI (full partition)")
    ax[0].plot(ns, [r["voronoi"] for r in rows], "^:", lw=2, ms=8, c="#34495e", label="Voronoi")
    ax[0].set_xlabel("vertex count $n_c$"); ax[0].set_ylabel("shape fidelity (IoU)")
    ax[0].set_title("Does $n_c$ control cell shape?"); ax[0].set_ylim(0, 1.02); ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot([c["margin"] for c in cov], [100 * c["coverage"] for c in cov], "o-", lw=2.5, ms=9, c="#2980b9")
    ax[1].set_xlabel("margin (mortar width)"); ax[1].set_ylabel("cell coverage (%)")
    ax[1].set_title("Mortar width is a direct dial"); ax[1].grid(alpha=.3)
    sol = [out["reachable"]["moroni"][f"{g:.2f}"]["median_solidity"] for g in gammas]
    dis = [out["reachable"]["moroni"][f"{g:.2f}"]["disconnected_pct"] for g in gammas]
    ax[2].plot(gammas, sol, "o-", lw=2.5, ms=9, c="#8e44ad", label="median solidity")
    ax[2].axhline(v["median_solidity"], ls="--", c="k", lw=2, label="Voronoi / any power diagram")
    a2 = ax[2].twinx(); a2.plot(gammas, dis, "s-", lw=2.5, ms=9, c="#c0392b", label="% disconnected")
    a2.set_ylabel("% cells disconnected", color="#c0392b")
    ax[2].set_xlabel("$\\gamma$ (phase weight)"); ax[2].set_ylabel("median cell solidity")
    ax[2].set_title("Leaving the weighted-diagram family"); ax[2].legend(loc="lower left", fontsize=11)
    ax[2].grid(alpha=.3)
    fig.tight_layout(); _save(fig, "control_quantitative.png")
    return out


# -------------------------------------------------------------------- bench
def cmd_bench():
    print("== bench ==")
    f = M.make_cells()
    out = {"machine": {"platform": platform.platform(), "numpy": np.__version__,
                       "python": sys.version.split()[0]},
           "config": {"n_cells": f.n, "rounds": ROUNDS}, "results": {}}
    for w, h in RESOLUTIONS:
        xv, yv = M.grid(w, h, f.bbox)
        br = M.BinnedRenderer(f); br.render(xv, yv)
        fns = {"voronoi_kdtree": lambda: M.voronoi_kdtree(f, xv, yv),
               "moroni_binned_exact": lambda: br.render(xv, yv),
               "moroni_reference": lambda: M.render_reference(f, xv, yv)}
        for g in fns.values():
            for _ in range(WARMUP): g()
        acc = {k: [] for k in fns}
        for _ in range(ROUNDS):
            for k, g in fns.items():
                gc.collect(); t0 = time.perf_counter(); g()
                acc[k].append((time.perf_counter() - t0) * 1e3)
        r = {k: {"median_ms": float(np.median(v)),
                 "iqr_ms": float(np.percentile(v, 75) - np.percentile(v, 25)),
                 "cv_pct": float(np.std(v) / np.mean(v) * 100)} for k, v in acc.items()}
        kd, mo, rf = (r[k]["median_ms"] for k in
                      ("voronoi_kdtree", "moroni_binned_exact", "moroni_reference"))
        r["ratio_vs_kdtree"] = kd / mo; r["speedup_vs_reference"] = rf / mo
        out["results"][f"{w}x{h}"] = r
        print(f"  {w}x{h}: kdtree={kd:7.1f}  moroni={mo:7.1f}  brute={rf:8.1f}  "
              f"{rf/mo:.1f}x vs brute, {kd/mo:.2f}x vs Voronoi")
    _dump(out, "bench.json")

    res = [f"{w}x{h}" for w, h in RESOLUTIONS]
    g = lambda k: [out["results"][r][k]["median_ms"] for r in res]
    kd, mo, rf = g("voronoi_kdtree"), g("moroni_binned_exact"), g("moroni_reference")
    x = np.arange(len(res))
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    ax[0].plot(x, kd, "o-", lw=2.5, ms=9, label="Voronoi (cKDTree)")
    ax[0].plot(x, mo, "^-", lw=2.5, ms=9, c="#27ae60", label="MORONI binned (exact)")
    ax[0].plot(x, rf, "s--", lw=2, ms=8, c="#c0392b", label="MORONI brute force")
    ax[0].set_yscale("log"); ax[0].set_ylabel("median runtime (ms, log scale)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(res); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Absolute runtime")
    ax[1].bar(x - .2, [a / b for a, b in zip(rf, mo)], .4, color="#27ae60", label="speedup vs brute force")
    ax[1].bar(x + .2, [a / b for a, b in zip(kd, mo)], .4, color="#34495e", label="ratio vs Voronoi")
    ax[1].axhline(1, ls="--", c="k", lw=2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(res); ax[1].set_ylabel("$\\times$")
    ax[1].legend(); ax[1].grid(alpha=.3, axis="y"); ax[1].set_title("Relative performance")
    fig.tight_layout(); _save(fig, "performance.png")
    return out


# ------------------------------------------------------------------ figures
def cmd_figures():
    print("== figures ==")
    f = M.make_cells(); xv, yv = M.grid(1100, 700, f.bbox)

    # 1. the bug, and the fix
    fa = M.make_cells(sides=(3, 4), radius=0.62, jitter=0.05, seed=7, rows=5, cols=7)
    gx, gy = M.grid(1000, 720, fa.bbox)
    raw_u = M.render_reference(fa, gx, gy, gamma=0.0)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    _show(ax[0], M.voronoi_kdtree(fa, gx, gy), "Voronoi\nno shape control")
    _show(ax[1], M.apply_margin(raw_u, fa, gx, gy, None), "Full partition\nhexagonal regardless of $n_c$")
    _show(ax[2], M.apply_margin(raw_u, fa, gx, gy, 0.0), "Signed field + margin\n$n_c=3$ gives triangles")
    fig.tight_layout(); _save(fig, "shape_control.png")

    # 2. energy decomposition
    c = f.centers
    dcs = np.hypot(xv[None] - c[:, 0][:, None, None], yv[None] - c[:, 1][:, None, None])
    i0 = np.argmin(dcs, 0)
    dx, dy = xv - c[i0, 0], yv - c[i0, 1]
    dc = np.take_along_axis(dcs, i0[None], 0)[0]
    rad = f.lut_radius[i0]; ln = f.luts.shape[-1]
    ix = np.clip(((dx + rad) * (ln - 1) / (2 * rad)).astype(np.int32), 0, ln - 1)
    iy = np.clip(((dy + rad) * (ln - 1) / (2 * rad)).astype(np.int32), 0, ln - 1)
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    for a, (img, t, cm) in zip(ax, [
            (dc, "$d_c$  (centre distance)", "magma"),
            (f.luts[i0, iy, ix], "$s_c$  (signed polygon SDF)", "coolwarm"),
            (M._phase_algebraic(dx, dy, dc, 1.0), "phase  $1-\\cos(2\\theta+\\lambda d)$", "magma"),
            (None, "argmin $E_c$", None)]):
        if img is None:
            _show(a, M.apply_margin(M.BinnedRenderer(f).render(xv, yv), f, xv, yv, 0.05), t)
        else:
            a.imshow(img, cmap=cm); a.set_title(t, fontsize=15); a.axis("off")
    fig.tight_layout(); _save(fig, "energy_decomposition.png")

    # 3. artist controls, one row each, large
    sweeps = [
        ("control_shape.png", "vertex count $n_c$ -- cell shape",
         [(f"$n_c$={v}", dict(sides=(v, v + 1))) for v in (3, 4, 5, 6, 8)], dict(margin=0.0)),
        ("control_size.png", "radius $r$ -- cell size",
         [(f"$r$={v:.2f}", dict(radius=v)) for v in (0.30, 0.42, 0.54, 0.66, 0.78)], dict(margin=0.02)),
        ("control_irregularity.png", "jitter $J$ -- boundary irregularity",
         [(f"$J$={v:.2f}", dict(jitter=v)) for v in (0.0, 0.15, 0.30, 0.45, 0.60)], dict(margin=0.02)),
    ]
    for name, title, cases, rkw in sweeps:
        fig, ax = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.4))
        for a, (lab, kw) in zip(ax, cases):
            base = dict(radius=0.6, jitter=0.15, seed=7); base.update(kw)
            fl = M.make_cells(**base)
            ggx, ggy = M.grid(620, 420, fl.bbox)
            _show(a, _render(fl, ggx, ggy, gamma=0.0, **rkw), lab)
        fig.suptitle(title, y=1.02); fig.tight_layout(); _save(fig, name)

    for name, title, key, vals, rkw in [
            ("control_mortar.png", "margin -- mortar / grout width", "margin",
             (-0.08, -0.03, 0.02, 0.08, 0.18), {}),
            ("control_fracture.png", "$\\gamma$ -- fracture intensity", "gamma",
             (0.0, 0.08, 0.16, 0.26, 0.35), dict(margin=0.05)),
            ("control_fidelity.png", "$\\beta$ -- shape fidelity", "beta",
             (0.0, 0.2, 0.5, 0.9, 1.4), dict(margin=0.02, gamma=0.0))]:
        fig, ax = plt.subplots(1, len(vals), figsize=(4.6 * len(vals), 4.4))
        fl = M.make_cells(radius=0.58, jitter=0.18, seed=7)
        ggx, ggy = M.grid(620, 420, fl.bbox)
        for a, v in zip(ax, vals):
            kw = dict(rkw); kw[key] = v
            _show(a, _render(fl, ggx, ggy, **kw), f"{v:+.2f}" if key == "margin" else f"{v:.2f}")
        fig.suptitle(title, y=1.02); fig.tight_layout(); _save(fig, name)

    # 4. presets a game dev would actually reach for
    presets = [("Cracked mud", dict(sides=(6, 9), radius=0.6, jitter=0.4), dict(gamma=0.30, margin=0.05)),
               ("Stone wall",  dict(sides=(4, 6), radius=0.62, jitter=0.12), dict(gamma=0.02, beta=0.9, margin=0.03)),
               ("Ceramic tile", dict(sides=(4, 5), radius=0.6, jitter=0.02), dict(gamma=0.0, beta=1.2, margin=0.0)),
               ("Shattered glass", dict(sides=(3, 5), radius=0.7, jitter=0.5), dict(gamma=0.35, margin=0.08))]
    fig, ax = plt.subplots(1, len(presets), figsize=(5.2 * len(presets), 4.8))
    for a, (nm, ck, rk) in zip(ax, presets):
        fl = M.make_cells(seed=7, **ck)
        ggx, ggy = M.grid(700, 470, fl.bbox)
        _show(a, _render(fl, ggx, ggy, **rk), nm)
    fig.tight_layout(); _save(fig, "presets.png")

    # 5. side by side
    fig, ax = plt.subplots(1, 2, figsize=(18, 5.8))
    _show(ax[0], M.voronoi_kdtree(f, xv, yv), "Voronoi")
    _show(ax[1], M.apply_margin(M.BinnedRenderer(f).render(xv, yv), f, xv, yv, 0.05), "MORONI")
    fig.tight_layout(); _save(fig, "voronoi_vs_moroni.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["verify", "control", "bench", "figures", "all"])
    c = ap.parse_args().command
    t0 = time.time()
    for k in (["verify", "control", "bench", "figures"] if c == "all" else [c]):
        globals()[f"cmd_{k}"]()
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
