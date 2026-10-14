"""
Rigorous evaluation of LUT accuracy and comparative baselines for MORONI paper.
Quantifies:
1. Signed Distance LUT error vs Analytical Exact Polygon Distance (MAE, RMSE, L_inf, and Label Mismatch %)
2. Comparative Baseline Analysis (Voronoi L2 unmasked, Voronoi L2 + Clipping, L1, Linf, Power Diagram, Apollonius, MORONI variants)
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
    
    # 1. Distance field error over 50 varied test polygons
    test_polys = []
    for _ in range(50):
        n = int(rng.integers(3, 9))
        r = float(rng.uniform(0.35, 0.75))
        jitter = float(rng.uniform(0.05, 0.45))
        ctr = np.zeros(2, dtype=np.float32)
        p = M._root_polygon(ctr, n, r, jitter, rng)
        test_polys.append((p, r))
    
    query_dim = 500
    w, h = 900, 560
    f_default = M.make_cells(seed=22, lut_size=72)
    xv, yv = M.grid(w, h, f_default.bbox)
    alpha, beta, gamma = 0.58, 0.32, 0.10

    # Compute exact analytical labels on default field once
    print("  Computing exact analytical ground truth field (900x560)...")
    best_v_exact = np.full(xv.shape, np.inf, np.float32)
    best_i_exact = np.zeros(xv.shape, np.int32)
    for i, (poly, ctr) in enumerate(zip(f_default.polys, f_default.centers)):
        dx, dy = xv - ctr[0], yv - ctr[1]
        dc = np.hypot(dx, dy)
        s_exact = exact_analytical_sdf(xv, yv, poly)
        ph = M._phase_atan2(dx, dy, dc, 2.0, 1.0)
        v = alpha * dc + beta * s_exact + gamma * ph
        m = v < best_v_exact
        best_v_exact[m], best_i_exact[m] = v[m], i

    for lsize in lut_sizes:
        maes, rmses, l_infs = [], [], []
        for poly, rad in test_polys:
            rr = float(np.max(np.linalg.norm(poly, axis=1)) * 1.5)
            # Create LUT
            ax_lut = np.linspace(-rr, rr, lsize, dtype=np.float32)
            lx, ly = np.meshgrid(ax_lut, ax_lut)
            lut = exact_analytical_sdf(lx, ly, poly)
            
            # Query at high-density grid
            qx = np.linspace(-rr, rr, query_dim, dtype=np.float32)
            qy = np.linspace(-rr, rr, query_dim, dtype=np.float32)
            qxv, qyv = np.meshgrid(qx, qy)
            
            exact = exact_analytical_sdf(qxv, qyv, poly)
            sampled = M._sample_lut(qxv, qyv, lut, rr, bilinear=True)
            
            diff = np.abs(sampled - exact)
            maes.append(float(np.mean(diff)))
            rmses.append(float(np.sqrt(np.mean(diff ** 2))))
            l_infs.append(float(np.max(diff)))
            
        # Compute label mismatch on field
        f_eval = M.make_cells(seed=22, lut_size=lsize)
        lbl_lut = M.render_reference(f_eval, xv, yv, alpha=alpha, beta=beta, gamma=gamma)
        label_mismatch = float(np.mean(lbl_lut != best_i_exact) * 100.0)

        results[f"lut_{lsize}"] = {
            "lut_size": lsize,
            "mean_mae": float(np.mean(maes)),
            "mean_rmse": float(np.mean(rmses)),
            "mean_l_inf": float(np.mean(l_infs)),
            "max_l_inf": float(np.max(l_infs)),
            "label_mismatch_pct": label_mismatch,
        }
        print(f"  LUT {lsize}x{lsize}: MAE={results[f'lut_{lsize}']['mean_mae']:.5f}, "
              f"RMSE={results[f'lut_{lsize}']['mean_rmse']:.5f}, "
              f"L_inf_max={results[f'lut_{lsize}']['max_l_inf']:.5f}, "
              f"Label Error={label_mismatch:.4f}%")
        
    return results

def eval_baselines():
    print("Evaluating Baselines...")
    f = M.make_cells(seed=22)
    w, h = 900, 560
    xv, yv = M.grid(w, h, f.bbox)
    pts = np.column_stack([xv.ravel(), yv.ravel()])
    c = f.centers
    
    # 1. Standard Voronoi (L2 Euclidean, unmasked)
    tree = cKDTree(c)
    lbl_vor_l2 = tree.query(pts, k=1, workers=-1)[1].reshape(h, w).astype(np.int32)
    
    # 2. Standard Voronoi + Polygon Clipping (margin=0.0)
    lbl_vor_l2_clip = M.apply_margin(lbl_vor_l2, f, xv, yv, margin=0.0)

    # 3. Manhattan Voronoi (L1)
    lbl_vor_l1 = tree.query(pts, k=1, p=1, workers=-1)[1].reshape(h, w).astype(np.int32)
    
    # 4. Chebyshev Voronoi (Linf)
    lbl_vor_linf = tree.query(pts, k=1, p=np.inf, workers=-1)[1].reshape(h, w).astype(np.int32)
    
    # 5. Power Diagram (Laguerre): argmin ||x - c_i||^2 - w_i
    rng = np.random.default_rng(22)
    weights = (rng.uniform(0.3, 0.8, size=len(c)) ** 2).astype(np.float32)
    dcs_sq = np.sum((pts[:, None, :] - c[None, :, :]) ** 2, axis=2) # (P, N)
    lbl_power = np.argmin(dcs_sq - weights[None, :], axis=1).reshape(h, w).astype(np.int32)
    
    # 6. Additively Weighted Voronoi (Apollonius): argmin ||x - c_i|| - w_i
    dcs = np.sqrt(dcs_sq)
    lbl_apollonius = np.argmin(dcs - (np.sqrt(weights))[None, :], axis=1).reshape(h, w).astype(np.int32)
    
    # 7. MORONI gamma=0.0 (no fracture, polygon SDF only) + Clipping
    lbl_moroni_g0 = M.apply_margin(M.render_reference(f, xv, yv, alpha=0.58, beta=0.32, gamma=0.0), f, xv, yv, margin=0.0)
    
    # 8. MORONI default gamma=0.10 + Clipping
    lbl_moroni_def = M.apply_margin(M.render_reference(f, xv, yv, alpha=0.58, beta=0.32, gamma=0.10), f, xv, yv, margin=0.0)
    
    # 9. MORONI fractured gamma=0.20 + Clipping
    lbl_moroni_f20 = M.apply_margin(M.render_reference(f, xv, yv, alpha=0.58, beta=0.32, gamma=0.20), f, xv, yv, margin=0.0)
    
    # 10. MORONI shattered gamma=0.35 + Clipping
    lbl_moroni_f35 = M.apply_margin(M.render_reference(f, xv, yv, alpha=0.58, beta=0.32, gamma=0.35), f, xv, yv, margin=0.0)
    
    generators = {
        "Voronoi (L2 Euclidean, unmasked)": lbl_vor_l2,
        "Voronoi (L2) + Clipping (m=0)": lbl_vor_l2_clip,
        "Voronoi (L1 Manhattan)": lbl_vor_l1,
        "Voronoi (L_inf Chebyshev)": lbl_vor_linf,
        "Power (Laguerre) Diagram": lbl_power,
        "Apollonius Diagram": lbl_apollonius,
        "MORONI (gamma=0.00, polygon only)": lbl_moroni_g0,
        "MORONI (gamma=0.10, default)": lbl_moroni_def,
        "MORONI (gamma=0.20, fractured)": lbl_moroni_f20,
        "MORONI (gamma=0.35, shattered)": lbl_moroni_f35,
    }
    
    def geom(lbl):
        comps, sol = [], []
        max_id = int(lbl.max())
        for i in range(max_id + 1):
            m = (lbl == i)
            if m.sum() < 50:
                continue
            # Filter components >= 4 pixels to ignore single-pixel discretization artifacts
            labeled_arr, num_feats = ndimage.label(m)
            counts = np.bincount(labeled_arr.ravel())[1:]
            valid_feats = int(np.sum(counts >= 4))
            comps.append(max(1, valid_feats))
            
            ys, xs = np.nonzero(m)
            try:
                # 4 corners per pixel to correctly calculate discrete raster convex hull area
                corners = np.vstack([
                    np.column_stack([xs - 0.5, ys - 0.5]),
                    np.column_stack([xs + 0.5, ys - 0.5]),
                    np.column_stack([xs + 0.5, ys + 0.5]),
                    np.column_stack([xs - 0.5, ys + 0.5]),
                ])
                ch = ConvexHull(corners)
                sol.append(float(np.clip(m.sum() / ch.volume, 0.0, 1.0)))
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
    print(f"\n{'Generator':34s} | {'Solidity':8s} | {'Non-Convex':10s} | {'Disconn %':9s}")
    print("-" * 70)
    for name, g_lbl in generators.items():
        stats = geom(g_lbl)
        res[name] = stats
        print(f"{name:34s} | {stats['median_solidity']:8.3f} | {stats['nonconvex_pct']:9.1f}% | {stats['disconnected_pct']:8.1f}%")
        
    return res

if __name__ == "__main__":
    lut_res = eval_lut_accuracy()
    base_res = eval_baselines()
    out = {"lut_accuracy": lut_res, "baseline_comparison": base_res}
    (RES / "extended_experiments.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {RES / 'extended_experiments.json'}")
