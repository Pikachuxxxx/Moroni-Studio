"""
Generate a 4-channel PBR texture map figure (Diffuse, Normal, Height, Roughness)
to show the practical output of MORONI Studio in the appendix.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import moroni as M

FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def generate():
    f = M.make_cells(sides=(4, 6), radius=0.62, jitter=0.12, seed=7)
    w, h = 600, 400
    xv, yv = M.grid(w, h, f.bbox)
    
    # Render with stone wall preset
    br = M.BinnedRenderer(f, beta=0.9)
    raw = br.render(xv, yv, beta=0.9, gamma=0.02)
    labels = M.apply_margin(raw, f, xv, yv, margin=0.03)
    
    # 1. Diffuse / Color Map
    diffuse = M.colorize(np.maximum(labels, 0))
    diffuse[labels < 0] = [0.12, 0.12, 0.12]
    
    # 2. Height Map (dome profile inside cell)
    ln = f.luts.shape[-1]
    safe_labels = np.maximum(labels, 0)
    dx = xv - f.centers[safe_labels, 0]
    dy = yv - f.centers[safe_labels, 1]
    rad = f.lut_radius[safe_labels]
    ix = np.clip(((dx + rad) * (ln - 1) / (2 * rad)).astype(np.int32), 0, ln - 1)
    iy = np.clip(((dy + rad) * (ln - 1) / (2 * rad)).astype(np.int32), 0, ln - 1)
    sdf = f.luts[safe_labels, iy, ix]
    
    height = np.clip(-sdf / 0.62, 0.0, 1.0)
    height = np.sqrt(height)
    height[labels < 0] = 0.0
    
    # 3. Normal Map via Sobel filter
    gy, gx = np.gradient(height * 3.5)
    normal = np.zeros((h, w, 3), dtype=np.float32)
    normal[..., 0] = -gx
    normal[..., 1] = -gy
    normal[..., 2] = 1.0
    norm_len = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    normal = normal / norm_len
    normal_rgb = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
    
    # 4. Roughness / Grout Mask
    roughness = np.full((h, w), 0.25, dtype=np.float32)
    roughness[labels < 0] = 0.95
    
    # Plot 1x4 panel
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(diffuse); axes[0].set_title("(a) Diffuse / Albedo Map", fontsize=14); axes[0].axis("off")
    axes[1].imshow(normal_rgb); axes[1].set_title("(b) Tangent Normal Map", fontsize=14); axes[1].axis("off")
    axes[2].imshow(height, cmap="gray"); axes[2].set_title("(c) Height / Displacement", fontsize=14); axes[2].axis("off")
    axes[3].imshow(roughness, cmap="magma"); axes[3].set_title("(d) Roughness / Grout Mask", fontsize=14); axes[3].axis("off")
    
    fig.tight_layout()
    out_path = FIG / "pbr_showcase.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate()
