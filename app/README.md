# MORONI Studio

Interactive companion to **MORONI: Polygon-Guided Cellular Textures via Roots-of-Unity Polygons and Phase-Modulated Boundaries**, by Phani Srikar.

Serve the repository root with `python -m http.server 8765 --bind 127.0.0.1`, then open http://127.0.0.1:8765/app/ . Serving the root also makes the validation fixture available at http://127.0.0.1:8765/app/validation.html . There is no JavaScript build step. The root `index.html` redirects to `app/` for static hosting.

The studio provides an unlit Paper Colors view, normal/height/roughness maps, energy-component views and a split comparison with Voronoi using the same sites and clipping rule. Shaded mode is removed. Twelve presets are visual starting points; their names do not imply physical material validation.

PNG export widths are 512, 1024 and 2048 pixels; height follows the lattice aspect ratio. The GPU renders each export directly with one sample per pixel and no wireframe overlay. The optional four-sample preview averages four offsets; it is not an analytical prefilter. Albedo contains no lighting. Normals use a square-root height and central differences at fixed world offsets, with positive green pointing down the image. Import data maps as linear data, and invert green if the target engine uses the opposite convention. Roughness is an illustrative 0.25 in cells and 0.95 in gaps. **Export Input Polygons** writes generating polygons, not final assignment contours.

The HUD samples GPU labels at 180 x 120. Solidity and 8-connected components are descriptive raster measurements, with a minimum component area of 0.0001 of the image. They do not prove continuous nonconvexity, disconnection or exclusion from a weighted-diagram family. GPU draw time comes from timer queries when supported; unavailable timing is shown explicitly. Display refresh FPS is separate from GPU execution time.

WebGL2 is the primary path. Basic CPU image/map fallback shares geometry, height and palette; its normals use raster differences and can differ from GPU exports. GPU diagnostic views, split comparison and GPU timing require WebGL2. Browser/device coverage is limited to the environment recorded in the validation report.

See [reproduction instructions](../REPRODUCIBILITY.md) for the test page, report import, measured configurations and paper build.
