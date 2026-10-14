# MORONI Studio — Interactive WebGL Cellular Texture Generator

An interactive companion application to the paper:
**"MORONI: Cellular Texture and Fracture Synthesis via Roots-of-Unity Polygons and Signed Boundary Fields"**
*Phani Srikar*

---

## Live Demo & Hosting

This app is a **zero-dependency, purely client-side WebGL2/HTML5 application**. It requires no build step, no Node.js/npm dependencies, and zero server infrastructure.

- **Live URL**: [https://pikachuxxxx.github.io/Moroni-Studio/](https://pikachuxxxx.github.io/Moroni-Studio/)

### 1. Instant Local Preview
To test or run locally:
```bash
cd app
python3 -m http.server 8080
```
Open [http://localhost:8080](http://localhost:8080) in any modern browser (Chrome, Safari, Firefox, Edge).

### 2. Hosting
- **GitHub Pages**:
  The `gh-pages` branch serves this directory directly at `https://pikachuxxxx.github.io/Moroni-Studio/`.

---

## Features

1. **Real-Time 60+ FPS GPU Synthesis**:
   - Implements the complete MORONI 3-term energy formulation directly in a GLSL fragment shader:
     $$E_c(\mathbf{x}) = \alpha d_c + \beta s_c(\mathbf{x}) + \gamma (1 - \cos(\kappa \theta_c + \lambda d_c))$$
   - Instant response to all 5 interactive art-direction dials (Shape $n_c$, Size $r$, Irregularity $J$, Grout $m$, and Fracture $\gamma$).

2. **A/B Split-Screen Comparison**:
   - Compare traditional Voronoi directly against MORONI side-by-side with an interactive draggable divider.

3. **Energy Component Decomposition**:
   - Inspect individual energy fields live: Center Distance $d_c$, Signed Polygon SDF $s_c$, and Angular Phase Wave.

4. **PBR Texture Studio & Exporter**:
   - Generates and exports production-ready textures up to 2K resolution:
     - **Diffuse / Color Map** (PNG)
     - **Tangent-Space Normal Map** (Sobel derivative filter, PNG)
     - **Height / Displacement Map** (Smooth dome interior SDF profile, PNG)
     - **Roughness / Grout Mask** (PNG)
     - **SVG Vector Contours** (Clean vector polygons)

5. **Live Geometric & Topological Metrics**:
   - Real-time HUD calculating median cell solidity, percentage of non-convex cells, and disconnected cell rate to continuously verify departure from Power (Laguerre) and Apollonius diagram families.
