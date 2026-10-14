import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt


OUT = Path("devlog3_runs")
OUT.mkdir(exist_ok=True)


def root_polygon(center, n_sides, radius, offset_range, rng):
    k = np.arange(n_sides)
    theta = 2 * np.pi * k / n_sides + rng.uniform(-offset_range, offset_range, n_sides)
    z = radius * np.exp(1j * theta)
    return np.c_[z.real, z.imag] + center


def build_polygons(rows, cols, spacing, radius, offset_range, rng):
    polys = []
    centers = []
    for r in range(rows):
        for c in range(cols):
            center = np.array([
                c * spacing + rng.uniform(-0.22, 0.22) * spacing,
                r * spacing + rng.uniform(-0.22, 0.22) * spacing,
            ])
            n_sides = int(rng.integers(5, 9))  # proper varied polygons, not squares
            p = root_polygon(center, n_sides, radius * rng.uniform(0.88, 1.10), offset_range, rng)
            centers.append(center)
            polys.append(p)
    return np.array(centers), polys


def interaction_relax(polys, centers, strength=0.08, iters=3):
    # lightweight polygon interaction: repel vertices from close foreign centers
    c = centers
    for _ in range(iters):
        for i, p in enumerate(polys):
            pi = p.copy()
            for j, cj in enumerate(c):
                if i == j:
                    continue
                v = pi - cj
                d = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
                mask = d[:, 0] < np.percentile(d[:, 0], 35)
                if np.any(mask):
                    pi[mask] += strength * (v[mask] / d[mask])
            polys[i] = 0.75 * polys[i] + 0.25 * pi
    return polys


def inset_polygon(poly, factor):
    c = poly.mean(axis=0)
    return c + (poly - c) * factor


def rasterize_polygons(polys, bbox, shape):
    h, w = shape
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, w)
    ys = np.linspace(ymin, ymax, h)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.column_stack([xv.ravel(), yv.ravel()])

    id_map = np.full((h, w), -1, dtype=np.int32)
    for i, p in enumerate(polys):
        if p.shape[0] < 3:
            continue
        mask = MplPath(p).contains_points(pts).reshape(h, w)
        id_map[(id_map < 0) & mask] = i
    return id_map


def fill_gaps(id_map, rng, mode="nearest"):
    filled = id_map.copy()
    hole = filled < 0
    if not np.any(hole):
        return filled
    if mode == "nearest":
        _, (iy, ix) = distance_transform_edt(hole, return_distances=True, return_indices=True)
        filled[hole] = filled[iy[hole], ix[hole]]
        return filled

    # random-neighbor
    h, w = filled.shape
    for _ in range(10):
        holes = np.argwhere(filled < 0)
        if holes.size == 0:
            break
        rng.shuffle(holes)
        for y, x in holes:
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            neigh = filled[y0:y1, x0:x1]
            vals = neigh[neigh >= 0]
            if vals.size:
                filled[y, x] = vals[rng.integers(0, vals.size)]

    if np.any(filled < 0):
        _, (iy, ix) = distance_transform_edt(filled < 0, return_distances=True, return_indices=True)
        filled[filled < 0] = filled[iy[filled < 0], ix[filled < 0]]
    return filled


def render_case(case_id, seed, offset_range, inset, gap_mode):
    rng = np.random.default_rng(seed)
    rows, cols = 9, 14
    spacing = 0.82
    radius = 0.56
    bbox = (-1.1, cols * spacing, -1.1, rows * spacing)

    centers, polys = build_polygons(rows, cols, spacing, radius, offset_range, rng)
    polys = interaction_relax(polys, centers, strength=0.06 + 0.02 * rng.random(), iters=4)

    fractured = [inset_polygon(p, inset * rng.uniform(0.93, 1.03)) for p in polys]

    id_map = rasterize_polygons(fractured, bbox=bbox, shape=(700, 1100))
    filled = fill_gaps(id_map, rng, mode=gap_mode)

    palette = np.array([
        [237, 230, 217], [223, 208, 188], [202, 186, 165],
        [181, 166, 146], [158, 145, 129], [136, 125, 111], [114, 106, 98]
    ], dtype=float) / 255.0
    colors = palette[rng.integers(0, len(palette), size=len(polys))]
    img = colors[filled]

    grain = rng.normal(0, 0.032, size=(img.shape[0], img.shape[1], 1))
    band = rng.normal(0, 0.012, size=(img.shape[0], 1, 1))
    img = np.clip(img + grain + band, 0, 1)

    fig = plt.figure(figsize=(11, 7), dpi=170)
    ax = fig.add_subplot(111)
    ax.imshow(img, origin="lower", extent=[bbox[0], bbox[1], bbox[2], bbox[3]])

    for p in fractured:
        pp = np.vstack([p, p[0]])
        ax.plot(pp[:, 0], pp[:, 1], color=(0.08, 0.07, 0.06, 0.32), linewidth=0.55)

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal")
    ax.axis("off")
    out_img = OUT / f"case_{case_id:02d}.png"
    plt.tight_layout(pad=0)
    fig.savefig(out_img, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    crack_ratio = float(np.mean(id_map < 0))
    unique_cells = int(len(np.unique(filled)))
    edge = (
        (filled != np.roll(filled, 1, axis=0)) |
        (filled != np.roll(filled, 1, axis=1))
    )
    boundary_density = float(edge.mean())
    return {
        "case": case_id,
        "seed": seed,
        "offset_range": offset_range,
        "inset": inset,
        "gap_mode": gap_mode,
        "crack_ratio": crack_ratio,
        "unique_cells": unique_cells,
        "boundary_density": boundary_density,
        "image": str(out_img),
    }


def main():
    runs = []
    cid = 1
    for seed in [11, 17, 23, 29]:
        for offset, inset, gap in [
            (0.35, 0.87, "nearest"),
            (0.52, 0.84, "random"),
            (0.62, 0.81, "random"),
        ]:
            runs.append(render_case(cid, seed, offset, inset, gap))
            cid += 1

    with open(OUT / "runs_summary.json", "w") as f:
        json.dump(runs, f, indent=2)

    # pick best by target boundary density range (devlog3-like seams)
    target = sorted(runs, key=lambda r: abs(r["boundary_density"] - 0.16))[:4]
    with open(OUT / "selected_cases.json", "w") as f:
        json.dump(target, f, indent=2)


if __name__ == "__main__":
    main()
