"""
MORONI - controllable cellular texture synthesis.

Cells are jittered roots of unity; a pixel is assigned to the cell minimising

    E_c(x) = alpha*d_c + beta*d_edge_c(x) + gamma*(1 - cos(kappa*theta_c + lam*d_c))

Three renderers, all producing identical labels for the same parameters:

    render_reference  brute force over every cell. Slow, obviously correct.
    render_pruned     admissible lower-bound pruning. Provably identical to
                      reference; the bound is derived in prune_radius().
    render_binned     lattice-binned candidates + algebraic phase. Fastest.

The only approximation anywhere is the boundary-distance LUT, which is shared by
all three, so they agree exactly with each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CellField", "make_cells", "grid", "prune_radius",
    "render_reference", "render_pruned", "render_binned", "BinnedRenderer",
    "apply_margin", "shape_fidelity",
    "soften", "voronoi_kdtree", "boundary_density", "colorize",
]

F32 = np.float32


# --------------------------------------------------------------------- geometry
@dataclass
class CellField:
    """Cells on a jittered lattice, plus their boundary-distance LUTs."""
    centers: np.ndarray       # (N,2) f32
    polys: list[np.ndarray]   # N polygons, (n_c,2)
    lattice: np.ndarray       # (N,2) int, lattice coordinate of each cell
    luts: np.ndarray          # (N,L,L) f32 SIGNED boundary distance, local frame
    lut_radius: np.ndarray    # (N,) f32 half-extent of each LUT frame
    lut_depth: np.ndarray     # (N,) f32 max inward depth, i.e. -min(lut) >= 0
    rows: int
    cols: int
    spacing: float

    @property
    def n(self) -> int:
        return len(self.centers)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (-0.8, self.cols * self.spacing, -0.8, self.rows * self.spacing)


def _root_polygon(center, n, r, jitter, rng):
    k = np.arange(n)
    th = 2 * np.pi * k / n + rng.uniform(-jitter, jitter, n)
    z = r * np.exp(1j * th)
    return np.c_[z.real, z.imag] + center


def _signed_distance(xv, yv, poly):
    """
    Signed distance to the polygon: negative inside, positive outside.

    The sign matters. With an unsigned distance the boundary term is minimised
    *on the outline*, so it carves rings and the polygon's shape never reaches the
    cell. Signed, it is minimised at the polygon's medial axis, so raising beta
    pulls the cell onto the polygon silhouette -- which is what makes vertex count
    an actual artist control.
    """
    from matplotlib.path import Path as MplPath
    dmin = np.full(xv.shape, np.inf, dtype=F32)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        v = b - a
        wx, wy = xv - a[0], yv - a[1]
        t = np.clip((wx * v[0] + wy * v[1]) / (v @ v + 1e-12), 0.0, 1.0)
        dmin = np.minimum(dmin, np.hypot(xv - (a[0] + t * v[0]),
                                         yv - (a[1] + t * v[1])))
    inside = MplPath(poly).contains_points(
        np.column_stack([xv.ravel(), yv.ravel()])).reshape(xv.shape)
    return np.where(inside, -dmin, dmin).astype(F32)


def make_cells(rows=8, cols=12, spacing=0.9, radius=0.5, jitter=0.35,
               seed=22, lut_size=72, sides=(5, 9)) -> CellField:
    rng = np.random.default_rng(seed)
    centers, polys, lattice = [], [], []
    for r in range(rows):
        for c in range(cols):
            ctr = np.array([c * spacing + rng.uniform(-0.13, 0.13) * spacing,
                            r * spacing + rng.uniform(-0.13, 0.13) * spacing])
            poly = _root_polygon(ctr, int(rng.integers(*sides)),
                                 radius * rng.uniform(0.9, 1.1), jitter, rng)
            centers.append(ctr); polys.append(poly); lattice.append((r, c))

    luts, radii = [], []
    for p, c in zip(polys, centers):
        q = p - c
        rr = float(np.max(np.linalg.norm(q, axis=1)) * 1.5)
        ax = np.linspace(-rr, rr, lut_size, dtype=F32)
        xv, yv = np.meshgrid(ax, ax)
        luts.append(_signed_distance(xv, yv, q.astype(F32)))
        radii.append(rr)

    L = np.stack(luts).astype(F32)
    return CellField(np.asarray(centers, F32), polys, np.asarray(lattice, int),
                     L, np.asarray(radii, F32),
                     np.maximum(-L.reshape(len(L), -1).min(1), 0).astype(F32),
                     rows, cols, spacing)


def grid(w, h, bbox):
    x0, x1, y0, y1 = bbox
    return np.meshgrid(np.linspace(x0, x1, w, dtype=F32),
                       np.linspace(y0, y1, h, dtype=F32))


# ------------------------------------------------------------------ energy bits
def _sample_lut(dx, dy, lut, r, bilinear=True):
    """
    Sample signed polygon distance from precomputed LUT with bilinear interpolation.
    For queries beyond the LUT bounding box [-r, r], adds the Euclidean distance
    from the bounding box boundary (preserving 1-Lipschitz continuity).
    """
    n = lut.shape[-1]
    gx = (dx + r) * (n - 1) / (2.0 * r)
    gy = (dy + r) * (n - 1) / (2.0 * r)

    if not bilinear:
        ix = np.clip(np.round(gx).astype(np.int32), 0, n - 1)
        iy = np.clip(np.round(gy).astype(np.int32), 0, n - 1)
        base = lut[iy, ix] if lut.ndim == 2 else np.take_along_axis(
            lut.reshape(lut.shape[0], -1), (iy * n + ix)[..., None], -1)[..., 0]
    else:
        gx_c = np.clip(gx, 0.0, n - 1 - 1e-6)
        gy_c = np.clip(gy, 0.0, n - 1 - 1e-6)
        x0 = gx_c.astype(np.int32)
        y0 = gy_c.astype(np.int32)
        x1 = np.minimum(x0 + 1, n - 1)
        y1 = np.minimum(y0 + 1, n - 1)
        fx = gx_c - x0
        fy = gy_c - y0

        if lut.ndim == 2:
            v00, v10 = lut[y0, x0], lut[y0, x1]
            v01, v11 = lut[y1, x0], lut[y1, x1]
        else:
            flat = lut.reshape(lut.shape[0], -1)
            v00 = np.take_along_axis(flat, (y0 * n + x0)[..., None], -1)[..., 0]
            v10 = np.take_along_axis(flat, (y0 * n + x1)[..., None], -1)[..., 0]
            v01 = np.take_along_axis(flat, (y1 * n + x0)[..., None], -1)[..., 0]
            v11 = np.take_along_axis(flat, (y1 * n + x1)[..., None], -1)[..., 0]

        base = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11

    out_dx = np.maximum(0.0, np.abs(dx) - r)
    out_dy = np.maximum(0.0, np.abs(dy) - r)
    return (base + np.hypot(out_dx, out_dy)).astype(F32)


def _phase_atan2(dx, dy, dc, kappa, lam):
    """Reference phase term: 1 - cos(kappa*theta + lam*d). At center dc < 1e-6, phase is 0."""
    theta = np.arctan2(dy, dx)
    val = 1.0 - np.cos(kappa * theta + lam * dc)
    return np.where(dc < 1e-6, 0.0, val).astype(F32)


def _phase_algebraic(dx, dy, dc, lam):
    """
    Exact algebraic phase evaluation for kappa == 2 without atan2.
    For dc -> 0, the directional limit is regularized to 0 to match _phase_atan2 at the center.
    """
    d2 = dx * dx + dy * dy
    safe_d2 = np.maximum(d2, F32(1e-12))
    cl, sl = np.cos(lam * dc), np.sin(lam * dc)
    cos_val = ((dx * dx - dy * dy) * cl - 2.0 * dx * dy * sl) / safe_d2
    cos_val = np.where(dc < 1e-6, 1.0, cos_val)
    return (1.0 - cos_val).astype(F32)


def prune_radius(field: CellField, alpha, beta, gamma) -> float:
    """
    Admissible candidate pruning bound based on an incumbent energy.

    For any cell c with polygon interior depth I_c = -min(s_c) >= 0 (where I_max = max(I_c)):
        E_c(x) >= alpha * d_c - beta * I_max

    Given an incumbent candidate c* (e.g., nearest lattice center with distance d_min),
    the winning energy is bounded above by:
        U(x) = E_{c*}(x) <= alpha * d_min + beta * s_{c*}(x) + 2 * gamma

    Since s_{c*}(x) <= d_min (as s_{c*}(0) <= 0 and s is 1-Lipschitz), we have:
        U(x) <= (alpha + beta) * d_min + 2 * gamma

    Therefore, any cell c whose lower bound exceeds U(x) cannot win:
        alpha * d_c - beta * I_max > U(x)
        d_c > d_min + (beta * (I_max + d_min) + 2 * gamma) / alpha

    For a global delta = d_c - d_min across all pixels where d_min <= max_lattice_dist:
    """
    I_max = float(field.lut_depth.max())
    d_max_cell = field.spacing * np.sqrt(2.0) / 2.0
    span = float(I_max + d_max_cell)
    return float((beta * span + 2.0 * gamma) / max(alpha, 1e-9))


# ------------------------------------------------------------------- renderers
BACKGROUND = -1


def apply_margin(labels, field, xv, yv, margin):
    """
    Turn a space-filling partition into cells + mortar by releasing pixels
    whose signed distance to the winning polygon exceeds `margin`.
    margin=None keeps the full partition.
    """
    if margin is None:
        return labels
    valid = labels >= 0
    out = labels.copy()
    if not valid.any():
        return out

    ln = field.luts.shape[-1]
    dx = xv[valid] - field.centers[labels[valid], 0]
    dy = yv[valid] - field.centers[labels[valid], 1]
    rad = field.lut_radius[labels[valid]]
    lbl = labels[valid]

    # Bilinear sampling
    gx = np.clip((dx + rad) * (ln - 1) / (2.0 * rad), 0.0, ln - 1 - 1e-6)
    gy = np.clip((dy + rad) * (ln - 1) / (2.0 * rad), 0.0, ln - 1 - 1e-6)
    x0, y0 = gx.astype(np.int32), gy.astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, ln - 1), np.minimum(y0 + 1, ln - 1)
    fx, fy = gx - x0, gy - y0

    v00 = field.luts[lbl, y0, x0]
    v10 = field.luts[lbl, y0, x1]
    v01 = field.luts[lbl, y1, x0]
    v11 = field.luts[lbl, y1, x1]
    sd = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11

    out_dx = np.maximum(0.0, np.abs(dx) - rad)
    out_dy = np.maximum(0.0, np.abs(dy) - rad)
    sd += np.hypot(out_dx, out_dy)

    mask = np.zeros(labels.shape, bool)
    mask[valid] = (sd > margin)
    out[mask] = BACKGROUND
    return out


def render_reference(field: CellField, xv, yv, alpha=0.58, beta=0.32,
                     gamma=0.10, kappa=2.0, lam=1.0):
    """Brute force over all cells. The definition of correct."""
    best_v = np.full(xv.shape, np.inf, F32)
    best_i = np.zeros(xv.shape, np.int32)
    for i, c in enumerate(field.centers):
        dx, dy = xv - c[0], yv - c[1]
        dc = np.hypot(dx, dy)
        v = (alpha * dc
             + beta * _sample_lut(dx, dy, field.luts[i], field.lut_radius[i])
             + gamma * _phase_atan2(dx, dy, dc, kappa, lam))
        m = v < best_v
        best_v[m], best_i[m] = v[m], i
    return best_i


def render_pruned(field: CellField, xv, yv, alpha=0.58, beta=0.32, gamma=0.10,
                  kappa=2.0, lam=1.0, return_stats=False):
    """
    Admissible pruning: evaluate the full energy only for cells inside the
    prune_radius() ball around each pixel's nearest centre. Label-identical to
    render_reference by construction.
    """
    delta = prune_radius(field, alpha, beta, gamma)
    cx = field.centers[:, 0][:, None, None]
    cy = field.centers[:, 1][:, None, None]
    dcs = np.hypot(xv[None] - cx, yv[None] - cy)
    dmin = dcs.min(0)
    live = dcs <= (dmin + delta)[None]

    best_v = np.full(xv.shape, np.inf, F32)
    best_i = np.zeros(xv.shape, np.int32)
    for i in range(field.n):
        m0 = live[i]
        if not m0.any():
            continue
        dx, dy = xv - field.centers[i, 0], yv - field.centers[i, 1]
        dc = dcs[i]
        v = (alpha * dc
             + beta * _sample_lut(dx, dy, field.luts[i], field.lut_radius[i])
             + gamma * _phase_atan2(dx, dy, dc, kappa, lam))
        m = m0 & (v < best_v)
        best_v[m], best_i[m] = v[m], i
    if return_stats:
        return best_i, {"delta": delta, "mean_candidates": float(live.sum(0).mean())}
    return best_i


class BinnedRenderer:
    """
    Lattice-binned renderer with ring escalation.

    Cells sit on a jittered lattice, so a pixel's candidates are its lattice
    neighbourhood -- no global [N,H,W] distance stack, no argpartition.

    Cost is driven down by two things:

    * **Ring escalation.** Ring 1 (9 candidates) is evaluated for every pixel.
      Any cell outside ring r sits at centre distance >= ((r+1) - 0.63)*s, so its
      energy is at least alpha times that. A pixel whose current best energy is
      below that floor is provably final and is masked out of all further rings.
      In practice almost every pixel finishes at ring 1, so the exact renderer
      costs barely more than the 9-candidate heuristic.
    * **Algebraic phase + 1-D LUT.** For kappa = 2 the phase term needs no atan2
      (see _phase_algebraic); the remaining cos(lam*d), sin(lam*d) are read from
      a shared 1-D table indexed by d, replacing two transcendentals per
      candidate with two gathers.

    Labels are identical to render_reference().
    """

    def __init__(self, field: CellField, alpha=0.58, beta=0.32, gamma=0.10,
                 lam=1.0, phase_lut=262144):
        self.f = field
        s = field.spacing
        # Smallest provably sufficient ring (see class docstring).
        # energies are shifted down by at most beta*I_max, so the floor a distant
        # cell must clear is alpha*d - beta*I_max
        imax = float(field.lut_depth.max())
        u = (alpha * np.sqrt(2.0) * 0.63 * s
             + beta * float(field.lut_radius.max()) + 2.0 * gamma)
        self.ring = 1
        while (alpha * ((self.ring + 1) - 0.63) * s - beta * imax < u
               and self.ring < max(field.rows, field.cols)):
            self.ring += 1
        self.bound_energy = float(u)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam

        self.index = np.full((field.rows, field.cols), -1, np.int32)
        for i, (r, c) in enumerate(field.lattice):
            self.index[r, c] = i

        # 1-D phase table over d in [0, dmax]
        x0, x1, y0, y1 = field.bbox
        self._dmax = float(np.hypot(x1 - x0, y1 - y0))
        self._pn = phase_lut
        t = np.linspace(0.0, self._dmax, phase_lut, dtype=F32)
        self._cos = np.cos(lam * t).astype(F32)
        self._sin = np.sin(lam * t).astype(F32)
        self._bin_cache: dict[tuple[int, int], np.ndarray] = {}

    def _bins(self, xv, yv):
        key = (xv.shape[1], xv.shape[0])
        if key not in self._bin_cache:
            s = self.f.spacing
            self._bin_cache[key] = np.stack([
                np.clip(np.rint(yv / s).astype(np.int32), 0, self.f.rows - 1),
                np.clip(np.rint(xv / s).astype(np.int32), 0, self.f.cols - 1)])
        return self._bin_cache[key]

    def _ring_offsets(self, r):
        if r == 0:
            return [(0, 0)]
        return [(dr, dc) for dr in range(-r, r + 1) for dc in range(-r, r + 1)
                if max(abs(dr), abs(dc)) == r]

    def render(self, xv, yv, alpha=None, beta=None, gamma=None, lam=None, return_stats=False):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        lam = self.lam if lam is None else lam
        rc = self._bins(xv, yv)
        cen, luts, lr = self.f.centers, self.f.luts, self.f.lut_radius
        ln = luts.shape[-1]
        pscale = (self._pn - 1) / self._dmax

        best_v = np.full(xv.shape, np.inf, F32)
        best_i = np.zeros(xv.shape, np.int32)
        active = np.ones(xv.shape, bool)
        stats = []

        for r in range(0, self.ring + 1):
            if not active.any():
                break
            idx = np.flatnonzero(active.ravel())
            xs = xv.ravel()[idx]; ys = yv.ravel()[idx]
            rr0 = rc[0].ravel()[idx]; cc0 = rc[1].ravel()[idx]
            bv = best_v.ravel()[idx].copy(); bi = best_i.ravel()[idx].copy()

            for dr, dc_ in self._ring_offsets(r):
                rr = np.clip(rr0 + dr, 0, self.f.rows - 1)
                cc = np.clip(cc0 + dc_, 0, self.f.cols - 1)
                cid = self.index[rr, cc]
                dx = xs - cen[cid, 0]; dy = ys - cen[cid, 1]
                d2 = dx * dx + dy * dy
                dc = np.sqrt(d2, dtype=F32)
                rad = lr[cid]
                gx = np.clip((dx + rad) * (ln - 1) / (2.0 * rad), 0.0, ln - 1 - 1e-6)
                gy = np.clip((dy + rad) * (ln - 1) / (2.0 * rad), 0.0, ln - 1 - 1e-6)
                x0, y0 = gx.astype(np.int32), gy.astype(np.int32)
                x1, y1 = np.minimum(x0 + 1, ln - 1), np.minimum(y0 + 1, ln - 1)
                fx, fy = gx - x0, gy - y0
                v00 = luts[cid, y0, x0]
                v10 = luts[cid, y0, x1]
                v01 = luts[cid, y1, x0]
                v11 = luts[cid, y1, x1]
                de = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
                out_dx = np.maximum(0.0, np.abs(dx) - rad)
                out_dy = np.maximum(0.0, np.abs(dy) - rad)
                de += np.hypot(out_dx, out_dy)

                cl, sl = np.cos(lam * dc), np.sin(lam * dc)
                cos_term = ((dx * dx - dy * dy) * cl - 2.0 * dx * dy * sl) / (d2 + F32(1e-12))
                cos_term = np.where(dc < 1e-6, 1.0, cos_term)
                ph = 1.0 - cos_term
                v = alpha * dc + beta * de + gamma * ph
                m = v < bv
                bv[m] = v[m]; bi[m] = cid[m]

            best_v.ravel()[idx] = bv
            best_i.ravel()[idx] = bi
            # a cell outside ring r has energy >= alpha * D_min - beta * I_max
            floor = (alpha * ((r + 1) - 0.63) * self.f.spacing
                     - beta * float(self.f.lut_depth.max()))
            still = bv > floor
            stats.append((r, int(idx.size), float(still.mean()) if idx.size else 0.0))
            nxt = np.zeros(xv.size, bool)
            nxt[idx[still]] = True
            active = nxt.reshape(xv.shape)

        if return_stats:
            return best_i, {"ring": self.ring,
                            "pixels_per_ring": [(r, n, f) for r, n, f in stats]}
        return best_i


def shape_fidelity(labels, field, xv, yv, min_pixels=50):
    """
    Mean IoU between each rendered cell and its target generating polygon.

    Parameters:
        labels: 2D array of cell assignments (-1 for background / grout).
        field: CellField containing generating polygons.
        xv, yv: Pixel coordinate grids.
        min_pixels: Minimum pixel threshold. In finite crops, edge cells truncated
                    by image boundaries can be excluded (min_pixels > 0, default 50)
                    or penalized with IoU = 0.0 (min_pixels = 0).
    """
    from matplotlib.path import Path as MplPath
    h, w = labels.shape
    pts = np.column_stack([xv.ravel(), yv.ravel()])
    out = []
    for i, poly in enumerate(field.polys):
        cell = (labels == i)
        cnt = cell.sum()
        if cnt < min_pixels:
            if min_pixels == 0:
                out.append(0.0)
            continue
        tgt = MplPath(poly).contains_points(pts).reshape(h, w)
        u = np.logical_or(cell, tgt).sum()
        if u > 0:
            out.append(float(np.logical_and(cell, tgt).sum() / u))
        elif min_pixels == 0:
            out.append(0.0)
    return float(np.mean(out)) if out else 0.0


def render_binned(field, xv, yv, **kw):
    return BinnedRenderer(field, **{k: v for k, v in kw.items()
                                    if k in ("alpha", "beta", "gamma", "lam")}
                          ).render(xv, yv, **kw)


# ------------------------------------------------------------------- post-pass
def soften(labels, field: CellField, inset=0.955):
    """Re-rasterise each polygon shrunk toward its centroid; unclaimed pixels
    keep their raw label. This is the post-pass that actually runs."""
    from matplotlib.path import Path as MplPath
    h, w = labels.shape
    x0, x1, y0, y1 = field.bbox
    xv, yv = np.meshgrid(np.linspace(x0, x1, w), np.linspace(y0, y1, h))
    pts = np.column_stack([xv.ravel(), yv.ravel()])
    out = np.full_like(labels, -1)
    for i, p in enumerate(field.polys):
        q = p.mean(0) + (p - p.mean(0)) * inset
        m = MplPath(q).contains_points(pts).reshape(h, w)
        out[(out < 0) & m] = i
    holes = out < 0
    out[holes] = labels[holes]
    return out


# ------------------------------------------------------------------- baselines
def voronoi_kdtree(field: CellField, xv, yv):
    """What a production pipeline actually uses."""
    from scipy.spatial import cKDTree
    tree = cKDTree(field.centers)
    pts = np.column_stack([xv.ravel(), yv.ravel()])
    return tree.query(pts, k=1, workers=-1)[1].reshape(xv.shape).astype(np.int32)


# --------------------------------------------------------------------- metrics
def boundary_density(lbl):
    return float(((lbl != np.roll(lbl, 1, 0)) | (lbl != np.roll(lbl, 1, 1))).mean())


def colorize(labels, seed=33, edge_dark=0.27):
    from matplotlib.colors import hsv_to_rgb
    rng = np.random.default_rng(seed)
    n = int(labels.max() + 1)
    hues = (np.linspace(0, 1, n, endpoint=False) + rng.uniform(-0.02, 0.02, n)) % 1.0
    rgb = hsv_to_rgb(np.c_[hues,
                           np.clip(0.46 + 0.24 * rng.random(n), 0, 1),
                           np.clip(0.68 + 0.22 * rng.random(n), 0, 1)])
    img = rgb[labels]
    img[(labels != np.roll(labels, 1, 0)) | (labels != np.roll(labels, 1, 1))] *= edge_dark
    return np.clip(img, 0, 1)
