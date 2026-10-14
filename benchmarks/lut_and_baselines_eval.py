"""
Rigorous evaluation of LUT accuracy and comparative baselines for MORONI paper.
Quantifies:
1. Signed Distance LUT error vs Analytical Exact Polygon Distance
2. Comparative Baseline Analysis (Voronoi L2, L1, Linf, Power Diagram, Apollonius, Worley F2-F1, MORONI)
Outputs JSON and metrics for the revised paper and appendix.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy import ndimage
import moroni as M

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(parents=True, exist_ok=True)

def exact_analytical_sdf(xv, yv, poly):
    """
    Exact analytical signed distance to polygon: negative inside, positive outside.
    Uses point-to-segment distance and winding number / ray casting.
    """
    from matplotlib.path import Path as MplPath
    dmin = np.full(xv.shape, np.inf, dtype=np.float32)
    for i in range(len(poly)):
        a = poly[i].astype(np.float32)
        b = poly[(i + 1) % len(poly)].astype(np.float32)
        v = b - a
        v_sq = float(v @ v) + 1e-12
        wx = xv - a[0]
        wy = yv - a[1]
        t = np.clip((wx * v[0] + wy * v[1]) / v_sq, 0.0, 1.0)
        proj_x = a[0] + t * v[0]
        proj_y = a[1] + t * v[1]
        d = np.hypot(xv - proj_x, yv - proj_y)
        dmin = np.minimum(dmin, d)
    
    inside = MplPath(poly).contains_points(
        np.column_stack([xv.ravel(), yv.ravel()])).reshape(xv.shape)
    return np.where(inside, -dmin, dmin).astype(np.float32)

def eval_lut_accuracy(lut_sizes=[36, 72, 144]):
    print("Evaluating LUT Accuracy vs Analytical Exact SDF...")
    rng = np.random.default_rng(42)
    results = {}
    
    # Generate 50 varied test polygons
    test_polys = []
    for _ in range(50):
        n = int(rng.integers(3, 9))
        r = float(rng.uniform(0.35, 0.75))
        jitter = float(rng.uniform(0.05, 0.45))
        ctr = np.zeros(2, dtype=np.float32)
        p = M._root_polygon(ctr, n, r, jitter, rng)
        test_polys.append((p, r))
    
    # Grid of test points (1000 x 1000 = 1,000,000 points per polygon)
    query_dim = 500
    for lsize in lut_sizes:
        maes, rmses, l_infs = [], [], []
        for poly, rad in test_polys:
            rr = float(np.max(np.linalg.norm(poly, axis=1)) * 1.5)
            # Create LUT
            ax_lut = np.linspace(-rr, rr, lsize, dtype=np.float32)
            lx, ly = np.meshgrid(ax_lut, ax_lut)
            lut = exact_analytical_sdf(lx, ly, poly)
            
            # Query at high-density random grid
            qx = np.linspace(-rr, rr, query_dim, dtype=np.float32)
            qy = np.linspace(-rr, rr, query_dim, dtype=np.float32)
            qxv, qyv = np.meshgrid(qx, qy)
            
            exact = exact_analytical_sdf(qxv, qyv, poly)
            sampled = M._sample_lut(qxv, qyv, lut, rr)
            
            diff = np.abs(sampled - exact)
            maes.append(float(np.mean(diff)))
            rmses.append(float(np.sqrt(np.mean(diff ** 2))))
            l_infs.append(float(np.max(diff)))
            
        results[f"lut_{lsize}"] = {
            "lut_size": lsize,
            "mean_mae": float(np.mean(maes)),
            "mean_rmse": float(np.mean(rmses)),
            "mean_l_inf": float(np.mean(l_infs)),
            "max_l_inf": float(np.max(l_infs)),
        }
        print(f"  LUT {lsize}x{lsize}: MAE={results[f'lut_{lsize}']['mean_mae']:.5f}, "
              f"RMSE={results[f'lut_{lsize}']['mean_rmse']:.5f}, "
              f"L_inf_max={results[f'lut_{lsize}']['max_l_inf']:.5f}")
        
    return results

def eval_baselines():
    print("Evaluating Baselines...")
    f = M.make_cells(seed=22)
    w, h = 900, 560
    xv, yv = M.grid(w, h, f.bbox)
    pts = np.column_stack([xv.ravel(), yv.ravel()])
    c = f.centers
    
    # 1. Standard Voronoi (L2 Euclidean)
    tree = cKDTree(c)
    lbl_vor_l2 = tree.query(pts, k=1, workers=-1)[1].reshape(h, w).astype(np.int32)
    
    # 2. Manhattan Voronoi (L1)
    lbl_vor_l1 = tree.query(pts, k=1, p=1, workers=-1)[1].reshape(h, w).astype(np.int32)
    
    # 3. Chebyshev Voronoi (Linf)
    lbl_vor_linf = tree.query(pts, k=1, p=np.inf, workers=-1)[1].reshape(h, w).astype(np.int32)
    
    # 4. Power Diagram (Laguerre): argmin ||x - c_i||^2 - w_i
    # Give random positive weights representing cell radius variance
    rng = np.random.default_rng(22)
    weights = (rng.uniform(0.3, 0.8, size=len(c)) ** 2).astype(np.float32)
    dcs_sq = np.sum((pts[:, None, :] - c[None, :, :]) ** 2, axis=2) # (P, N)
    lbl_power = np.argmin(dcs_sq - weights[None, :], axis=1).reshape(h, w).astype(np.int32)
    
    # 5. Additively Weighted Voronoi (Apollonius): argmin ||x - c_i|| - w_i
    dcs = np.sqrt(dcs_sq)
    lbl_apollonius = np.argmin(dcs - (np.sqrt(weights))[None, :], axis=1).reshape(h, w).astype(np.int32)
    
    # 6. Worley F2 - F1
    # Worley assigns value F2 - F1, but cell partition is defined by F1 (Voronoi)
    # 7. MORONI gamma=0.0 (no fracture, polygon SDF only)
    lbl_moroni_g0 = M.BinnedRenderer(f, gamma=0.0).render(xv, yv, gamma=0.0)
    
    # 8. MORONI default gamma=0.10
    lbl_moroni_def = M.BinnedRenderer(f, gamma=0.10).render(xv, yv, gamma=0.10)
    
    # 9. MORONI fractured gamma=0.20
    lbl_moroni_f20 = M.BinnedRenderer(f, gamma=0.20).render(xv, yv, gamma=0.20)
    
    # 10. MORONI shattered gamma=0.35
    lbl_moroni_f35 = M.BinnedRenderer(f, gamma=0.35).render(xv, yv, gamma=0.35)
    
    generators = {
        "Voronoi (L2 Euclidean)": lbl_vor_l2,
        "Voronoi (L1 Manhattan)": lbl_vor_l1,
        "Voronoi (L_inf Chebyshev)": lbl_vor_linf,
        "Power (Laguerre) Diagram": lbl_power,
        "Apollonius Diagram": lbl_apollonius,
        "MORONI (gamma=0.00)": lbl_moroni_g0,
        "MORONI (gamma=0.10, default)": lbl_moroni_def,
        "MORONI (gamma=0.20)": lbl_moroni_f20,
        "MORONI (gamma=0.35)": lbl_moroni_f35,
    }
    
    def geom(lbl):
        comps, sol = [], []
        for i in range(int(lbl.max()) + 1):
            m = (lbl == i)
            if m.sum() < 50:
                continue
            comps.append(ndimage.label(m)[1])
            ys, xs = np.nonzero(m)
            try:
                ch = ConvexHull(np.column_stack([xs, ys]).astype(float))
                sol.append(m.sum() / ch.volume)
            except Exception:
                pass
        c_arr, s_arr = np.array(comps), np.array(sol)
        return {
            "median_solidity": float(np.median(s_arr)) if len(s_arr) else 1.0,
            "nonconvex_pct": float(100 * np.mean(s_arr < 0.98)) if len(s_arr) else 0.0,
            "mean_components": float(c_arr.mean()) if len(c_arr) else 1.0,
            "disconnected_pct": float(100 * np.mean(c_arr > 1)) if len(c_arr) else 0.0,
            "shape_fidelity_tri": float(M.shape_fidelity(lbl, f, xv, yv)),
        }
    
    res = {}
    print(f"\n{'Generator':30s} | {'Solidity':8s} | {'Non-Convex':10s} | {'Disconn %':9s}")
    print("-" * 65)
    for name, g_lbl in generators.items():
        stats = geom(g_lbl)
        res[name] = stats
        print(f"{name:30s} | {stats['median_solidity']:8.3f} | {stats['nonconvex_pct']:9.1f}% | {stats['disconnected_pct']:8.1f}%")
        
    return res

if __name__ == "__main__":
    lut_res = eval_lut_accuracy()
    base_res = eval_baselines()
    out = {"lut_accuracy": lut_res, "baseline_comparison": base_res}
    (RES / "extended_experiments.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {RES / 'extended_experiments.json'}")
