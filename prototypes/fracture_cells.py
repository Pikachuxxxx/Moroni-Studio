import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt


def generate_root_polygon(n_sides, center, scale, offset_range, rng):
    """Polygon from roots of unity with angular jitter: z_k = scale * exp(i*theta_k)."""
    base = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    jitter = rng.uniform(-offset_range, offset_range, size=n_sides)
    theta = base + jitter
    z = scale * np.exp(1j * theta)
    poly = np.column_stack([z.real, z.imag]) + center
    return poly


def build_cells(rows, cols, spacing, n_sides, scale, offset_range, seed):
    rng = np.random.default_rng(seed)
    centers, polys = [], []
    for r in range(rows):
        for c in range(cols):
            center = np.array([
                c * spacing + rng.uniform(-0.25, 0.25) * spacing,
                r * spacing + rng.uniform(-0.25, 0.25) * spacing,
            ])
            poly = generate_root_polygon(n_sides, center, scale, offset_range, rng)
            centers.append(center)
            polys.append(poly)
    return np.array(centers), polys


def point_to_polyline_distance(xv, yv, poly):
    dmin = np.full(xv.shape, np.inf)
    for i in range(len(poly)):
        p0 = poly[i]
        p1 = poly[(i + 1) % len(poly)]
        vx, vy = p1[0] - p0[0], p1[1] - p0[1]
        wx, wy = xv - p0[0], yv - p0[1]
        denom = vx * vx + vy * vy + 1e-12
        t = np.clip((wx * vx + wy * vy) / denom, 0.0, 1.0)
        px = p0[0] + t * vx
        py = p0[1] + t * vy
        d = np.hypot(xv - px, yv - py)
        dmin = np.minimum(dmin, d)
    return dmin


def root_field_labels(centers, polys, xv, yv, alpha=0.62, beta=0.38, phase_warp=0.12):
    """
    Pure root-unity pseudo-cells (no Voronoi):
      E_k = alpha * d(center_k) + beta * d(polygon_edges_k) + phase term from exp(i*theta).
    """
    fields = []
    for c, p in zip(centers, polys):
        dc = np.hypot(xv - c[0], yv - c[1])
        de = point_to_polyline_distance(xv, yv, p)

        # e^{i theta} phase-like warp around each center to mimic natural fracture drift.
        theta = np.arctan2(yv - c[1], xv - c[0])
        phase = 1.0 - np.real(np.exp(1j * (3.0 * theta + 2.0 * dc)))

        f = alpha * dc + beta * de + phase_warp * phase
        fields.append(f)

    return np.argmin(np.stack(fields), axis=0)


def erode_per_cell(id_map, erosion_px, rng):
    """Create fracture gaps by shrinking each region independently."""
    h, w = id_map.shape
    fractured = np.full((h, w), -1, dtype=np.int32)
    n_cells = id_map.max() + 1

    for cid in range(n_cells):
        mask = id_map == cid
        if not np.any(mask):
            continue
        # randomize erosion per cell for natural crack width variation
        local_erode = max(1, int(erosion_px + rng.integers(-2, 3)))
        dist = distance_transform_edt(mask)
        keep = dist > local_erode
        fractured[keep] = cid

    return fractured


def fill_gaps_nearest(id_map):
    filled = id_map.copy()
    hole = filled < 0
    if not np.any(hole):
        return filled
    _, (iy, ix) = distance_transform_edt(hole, return_distances=True, return_indices=True)
    filled[hole] = filled[iy[hole], ix[hole]]
    return filled


def fill_gaps_random_neighbor(id_map, rng, rounds=8):
    filled = id_map.copy()
    h, w = filled.shape
    for _ in range(rounds):
        holes = np.argwhere(filled < 0)
        if len(holes) == 0:
            return filled
        rng.shuffle(holes)
        for y, x in holes:
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            vals = filled[y0:y1, x0:x1]
            vals = vals[vals >= 0]
            if vals.size:
                filled[y, x] = vals[rng.integers(0, vals.size)]
    return fill_gaps_nearest(filled)


def render(out_path, gap_fill_mode="nearest", seed=7):
    rng = np.random.default_rng(seed)

    # domain
    rows, cols = 8, 12
    spacing = 1.0
    n_sides = 6
    scale = 0.47
    offset_range = 0.55

    width, height = 560, 360
    xmin, xmax = -1.0, cols * spacing
    ymin, ymax = -1.0, rows * spacing

    centers, polys = build_cells(rows, cols, spacing, n_sides, scale, offset_range, seed)

    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
    xv, yv = np.meshgrid(xs, ys)

    labels = root_field_labels(centers, polys, xv, yv, alpha=0.58, beta=0.42, phase_warp=0.14)

    fractured = erode_per_cell(labels, erosion_px=6, rng=rng)

    if gap_fill_mode == "nearest":
        filled = fill_gaps_nearest(fractured)
    elif gap_fill_mode == "random-neighbor":
        filled = fill_gaps_random_neighbor(fractured, rng)
    else:
        raise ValueError("gap_fill_mode must be nearest or random-neighbor")

    palette = np.array([
        [239, 232, 220], [224, 210, 191], [203, 189, 169],
        [181, 168, 149], [158, 147, 131], [135, 126, 113], [112, 104, 95],
    ], dtype=float) / 255.0

    cell_colors = palette[rng.integers(0, len(palette), size=labels.max() + 1)]
    image = cell_colors[filled]

    # paper grain + directional tear streaks
    grain = rng.normal(0.0, 0.03, size=(height, width, 1))
    streak = rng.normal(0.0, 0.01, size=(height, 1, 1))
    image = np.clip(image + grain + streak, 0.0, 1.0)

    fig = plt.figure(figsize=(12, 7), dpi=180)
    ax = fig.add_subplot(111)
    ax.imshow(image, origin="lower", extent=[xmin, xmax, ymin, ymax])

    for p in polys:
        pp = np.vstack([p, p[0]])
        ax.plot(pp[:, 0], pp[:, 1], color=(0.07, 0.06, 0.05, 0.20), linewidth=0.45)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    render("fracture_root_unity_nearest.png", gap_fill_mode="nearest", seed=18)
    render("fracture_root_unity_random.png", gap_fill_mode="random-neighbor", seed=21)
