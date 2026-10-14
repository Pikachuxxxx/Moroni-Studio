/**
 * MORONI Core JavaScript Engine
 * Mathematical implementation of MORONI, Voronoi, Power Diagrams, Apollonius, and Worley noise.
 * Includes topological metrics (Solidity, Non-Convexity, Disconnection) and PBR Texture Map Exporters.
 */

class MoroniCore {
    // Generate jittered roots of unity polygon
    static generateRootPolygon(center, n, r, jitter, prng) {
        const poly = [];
        for (let k = 0; k < n; k++) {
            let th = 2 * Math.PI * k / n + (prng() * 2 - 1) * jitter;
            let r_actual = r * (0.92 + 0.16 * prng());
            poly.push([
                center[0] + r_actual * Math.cos(th),
                center[1] + r_actual * Math.sin(th)
            ]);
        }
        return poly;
    }

    // Signed distance from point (x, y) to polygon
    // Negative inside, positive outside
    static pointPolygonSignedDistance(x, y, poly) {
        let dmin = Infinity;
        let inside = false;
        const n = poly.length;

        for (let i = 0, j = n - 1; i < n; j = i++) {
            const xi = poly[i][0], yi = poly[i][1];
            const xj = poly[j][0], yj = poly[j][1];

            // Point-in-polygon ray casting
            const intersect = ((yi > y) !== (yj > y)) &&
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi);
            if (intersect) inside = !inside;

            // Distance to segment [poly[j], poly[i]]
            const vx = xi - xj, vy = yi - yj;
            const wx = x - xj, wy = y - yj;
            const vsq = vx * vx + vy * vy + 1e-12;
            const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / vsq));
            const projx = xj + t * vx;
            const projy = yj + t * vy;
            const dist = Math.hypot(x - projx, y - projy);
            if (dist < dmin) dmin = dist;
        }

        return inside ? -dmin : dmin;
    }

    // Pseudo-random number generator (Mulberry32)
    static createPRNG(seed) {
        let s = Math.floor(seed) >>> 0;
        return function() {
            s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    // Build lattice cell field
    static buildCellField(params) {
        const { rows, cols, spacing, radius, jitter, sides, seed } = params;
        const prng = MoroniCore.createPRNG(seed);
        const centers = [];
        const polys = [];
        const weights = []; // for power/apollonius diagrams

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const cx = c * spacing + (prng() * 2 - 1) * (0.13 * spacing);
                const cy = r * spacing + (prng() * 2 - 1) * (0.13 * spacing);
                const center = [cx, cy];

                let n_sides = sides;
                if (!sides || sides === 0) {
                    n_sides = 5 + Math.floor(prng() * 4); // 5 to 8
                }
                const poly = MoroniCore.generateRootPolygon(center, n_sides, radius, jitter, prng);

                centers.push(center);
                polys.push(poly);
                weights.push(Math.pow(radius * (0.8 + 0.4 * prng()), 2));
            }
        }

        return { rows, cols, spacing, centers, polys, weights };
    }

    // Render CPU image buffer with requested generator
    static renderCPU(field, width, height, params) {
        const { generator, alpha, beta, gamma, kappa, lam, margin, useMargin } = params;
        const { rows, cols, spacing, centers, polys, weights } = field;

        const xmin = -0.8, xmax = cols * spacing + 0.8;
        const ymin = -0.8, ymax = rows * spacing + 0.8;

        const labels = new Int32Array(width * height);
        const distances = new Float32Array(width * height);
        const sdfs = new Float32Array(width * height);

        const nCells = centers.length;

        for (let py = 0; py < height; py++) {
            const y = ymin + (py / (height - 1)) * (ymax - ymin);
            const rowOffset = py * width;

            for (let px = 0; px < width; px++) {
                const x = xmin + (px / (width - 1)) * (xmax - xmin);
                const pIdx = rowOffset + px;

                let bestId = 0;
                let bestVal = Infinity;
                let bestDc = 0;
                let bestSc = 0;

                for (let i = 0; i < nCells; i++) {
                    const cx = centers[i][0], cy = centers[i][1];
                    const dx = x - cx, dy = y - cy;
                    const dc = Math.hypot(dx, dy);

                    let val = 0;

                    if (generator === 'voronoi_l2') {
                        val = dc;
                    } else if (generator === 'voronoi_l1') {
                        val = Math.abs(dx) + Math.abs(dy);
                    } else if (generator === 'voronoi_linf') {
                        val = Math.max(Math.abs(dx), Math.abs(dy));
                    } else if (generator === 'power') {
                        val = dc * dc - weights[i];
                    } else if (generator === 'apollonius') {
                        val = dc - Math.sqrt(weights[i]);
                    } else {
                        // MORONI
                        const sc = MoroniCore.pointPolygonSignedDistance(x, y, polys[i]);
                        // Algebraic phase for kappa=2
                        let ph = 0;
                        if (Math.abs(kappa - 2.0) < 0.05) {
                            const d2 = dx * dx + dy * dy + 1e-12;
                            ph = 1.0 - ((dx * dx - dy * dy) * Math.cos(lam * dc) - 2.0 * dx * dy * Math.sin(lam * dc)) / d2;
                        } else {
                            const th = Math.atan2(dy, dx);
                            ph = 1.0 - Math.cos(kappa * th + lam * dc);
                        }
                        val = alpha * dc + beta * sc + gamma * ph;

                        if (val < bestVal) {
                            bestVal = val;
                            bestId = i;
                            bestDc = dc;
                            bestSc = sc;
                        }
                        continue;
                    }

                    if (val < bestVal) {
                        bestVal = val;
                        bestId = i;
                        bestDc = dc;
                    }
                }

                // Check margin / grout release
                if (generator === 'moroni' && useMargin) {
                    if (bestSc > margin) {
                        labels[pIdx] = -1; // Background / grout
                    } else {
                        labels[pIdx] = bestId;
                    }
                } else {
                    labels[pIdx] = bestId;
                }

                distances[pIdx] = bestDc;
                sdfs[pIdx] = bestSc;
            }
        }

        return { labels, distances, sdfs, width, height };
    }

    // Compute topological and geometric metrics from label buffer
    static computeMetrics(labels, width, height, nCells) {
        // Find cell boundaries, area, convex hull and solidity
        const cellPixels = {};
        for (let i = 0; i < nCells; i++) cellPixels[i] = [];

        for (let y = 0; y < height; y++) {
            const rowOffset = y * width;
            for (let x = 0; x < width; x++) {
                const id = labels[rowOffset + x];
                if (id >= 0 && cellPixels[id]) {
                    cellPixels[id].push([x, y]);
                }
            }
        }

        let nonConvexCount = 0;
        let disconnectedCount = 0;
        let validCells = 0;
        const solidities = [];

        for (let i = 0; i < nCells; i++) {
            const pts = cellPixels[i];
            if (pts.length < 50) continue;
            validCells++;

            // Convex hull area approximation
            const chArea = MoroniCore.convexHullArea(pts);
            const solidity = pts.length / Math.max(1, chArea);
            solidities.push(solidity);

            if (solidity < 0.98) {
                nonConvexCount++;
            }

            // Check connectivity using flood fill
            const comps = MoroniCore.countConnectedComponents(pts, width, height);
            if (comps > 1) {
                disconnectedCount++;
            }
        }

        solidities.sort((a, b) => a - b);
        const medianSolidity = solidities.length ? solidities[Math.floor(solidities.length / 2)] : 1.0;

        return {
            validCells,
            medianSolidity: Number(medianSolidity.toFixed(3)),
            nonConvexPct: Number(((nonConvexCount / Math.max(1, validCells)) * 100).toFixed(1)),
            disconnectedPct: Number(((disconnectedCount / Math.max(1, validCells)) * 100).toFixed(1))
        };
    }

    // 2D Convex Hull Area (Graham scan / monotone chain)
    static convexHullArea(points) {
        if (points.length < 3) return points.length;

        // Subsample for fast computation if > 2000 points
        let sample = points;
        if (points.length > 1000) {
            const step = Math.ceil(points.length / 1000);
            sample = points.filter((_, idx) => idx % step === 0);
        }

        sample.sort((a, b) => a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]);

        const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

        const lower = [];
        for (const p of sample) {
            while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
                lower.pop();
            }
            lower.push(p);
        }

        const upper = [];
        for (let i = sample.length - 1; i >= 0; i--) {
            const p = sample[i];
            while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
                upper.pop();
            }
            upper.push(p);
        }

        lower.pop();
        upper.pop();
        const hull = lower.concat(upper);

        // Shoelace formula for polygon area
        let area = 0;
        let perimeter = 0;
        for (let i = 0; i < hull.length; i++) {
            const j = (i + 1) % hull.length;
            area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1];
            perimeter += Math.hypot(hull[j][0] - hull[i][0], hull[j][1] - hull[i][1]);
        }
        // Pick's theorem for discrete raster grid points in polygon: A_discrete = continuous_area + perimeter / 2 + 1
        const continuousArea = Math.abs(area) * 0.5;
        const discreteHullPoints = continuousArea + perimeter * 0.5 + 1.0;
        return Math.max(discreteHullPoints, points.length);
    }

    // Count connected components via 8-connected BFS
    static countConnectedComponents(points, width, height) {
        const pointSet = new Set(points.map(p => `${p[0]},${p[1]}`));
        let components = 0;

        for (const p of points) {
            const key = `${p[0]},${p[1]}`;
            if (!pointSet.has(key)) continue;

            components++;
            const queue = [p];
            pointSet.delete(key);

            while (queue.length > 0) {
                const [cx, cy] = queue.pop();
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const nkey = `${cx + dx},${cy + dy}`;
                        if (pointSet.has(nkey)) {
                            pointSet.delete(nkey);
                            queue.push([cx + dx, cy + dy]);
                        }
                    }
                }
            }
        }
        return components;
    }

    // Generate PBR Textures (Diffuse, Normal Map, Height Map, Roughness Map)
    static generatePBRMaps(renderResult, radius) {
        const { labels, sdfs, width, height } = renderResult;
        const total = width * height;

        const diffuseData = new Uint8ClampedArray(total * 4);
        const normalData = new Uint8ClampedArray(total * 4);
        const heightData = new Uint8ClampedArray(total * 4);
        const roughnessData = new Uint8ClampedArray(total * 4);

        // Palette generator
        const colors = [];
        for (let i = 0; i < 256; i++) {
            const h = (i * 137.5) % 360;
            const rgb = MoroniCore.hsvToRgb(h / 360, 0.65, 0.85);
            colors.push(rgb);
        }

        // 1. Height map
        const heightMap = new Float32Array(total);
        for (let i = 0; i < total; i++) {
            const lbl = labels[i];
            if (lbl < 0) {
                heightMap[i] = 0.0;
            } else {
                const sc = sdfs[i];
                let h = Math.max(0, -sc / radius);
                heightMap[i] = Math.sqrt(Math.min(1.0, h));
            }
        }

        // 2. Normal Map via Sobel operator on heightMap
        for (let y = 0; y < height; y++) {
            const y0 = Math.max(0, y - 1) * width;
            const y1 = y * width;
            const y2 = Math.min(height - 1, y + 1) * width;

            for (let x = 0; x < width; x++) {
                const x0 = Math.max(0, x - 1);
                const x1 = x;
                const x2 = Math.min(width - 1, x + 1);

                const idx = y1 + x1;
                const pixelIdx = idx * 4;

                // Sobel kernel for dH/dx and dH/dy
                const tl = heightMap[y0 + x0], t = heightMap[y0 + x1], tr = heightMap[y0 + x2];
                const l  = heightMap[y1 + x0],                        r  = heightMap[y1 + x2];
                const bl = heightMap[y2 + x0], b = heightMap[y2 + x1], br = heightMap[y2 + x2];

                const dX = (tr + 2 * r + br) - (tl + 2 * l + bl);
                const dY = (bl + 2 * b + br) - (tl + 2 * t + tr);

                // Normal vector [-1, 1] -> [0, 255]
                const strength = 3.5;
                const nz = 1.0;
                const len = Math.hypot(-dX * strength, -dY * strength, nz);
                const nx = (-dX * strength) / len;
                const ny = (-dY * strength) / len;
                const nz_norm = nz / len;

                normalData[pixelIdx]     = Math.floor((nx * 0.5 + 0.5) * 255);
                normalData[pixelIdx + 1] = Math.floor((ny * 0.5 + 0.5) * 255);
                normalData[pixelIdx + 2] = Math.floor((nz_norm * 0.5 + 0.5) * 255);
                normalData[pixelIdx + 3] = 255;

                // Height map RGB
                const hVal = Math.floor(heightMap[idx] * 255);
                heightData[pixelIdx]     = hVal;
                heightData[pixelIdx + 1] = hVal;
                heightData[pixelIdx + 2] = hVal;
                heightData[pixelIdx + 3] = 255;

                // Roughness & Grout Map
                const isGrout = labels[idx] < 0;
                const rVal = isGrout ? 245 : 60; // Grout is rough, tiles are smooth
                roughnessData[pixelIdx]     = rVal;
                roughnessData[pixelIdx + 1] = rVal;
                roughnessData[pixelIdx + 2] = rVal;
                roughnessData[pixelIdx + 3] = 255;

                // Diffuse / Color Map
                if (isGrout) {
                    diffuseData[pixelIdx]     = 25;
                    diffuseData[pixelIdx + 1] = 25;
                    diffuseData[pixelIdx + 2] = 25;
                } else {
                    const c = colors[labels[idx] % 256];
                    // Edge darkening
                    const isEdge = (labels[y1 + x0] !== labels[idx] || labels[y0 + x1] !== labels[idx]);
                    const edgeFactor = isEdge ? 0.35 : 1.0;
                    const shade = (0.75 + 0.35 * heightMap[idx]) * edgeFactor;

                    diffuseData[pixelIdx]     = Math.floor(c[0] * shade);
                    diffuseData[pixelIdx + 1] = Math.floor(c[1] * shade);
                    diffuseData[pixelIdx + 2] = Math.floor(c[2] * shade);
                }
                diffuseData[pixelIdx + 3] = 255;
            }
        }

        return {
            diffuse: new ImageData(diffuseData, width, height),
            normal: new ImageData(normalData, width, height),
            height: new ImageData(heightData, width, height),
            roughness: new ImageData(roughnessData, width, height)
        };
    }

    // Convert HSV to RGB [0-255]
    static hsvToRgb(h, s, v) {
        let r, g, b;
        const i = Math.floor(h * 6);
        const f = h * 6 - i;
        const p = v * (1 - s);
        const q = v * (1 - f * s);
        const t = v * (1 - (1 - f) * s);
        switch (i % 6) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            case 5: r = v; g = p; b = q; break;
        }
        return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
    }

    // Export SVG vector boundaries and polygons
    static exportSVG(field, width, height) {
        const { cols, rows, spacing, polys } = field;
        const xmin = -0.8, xmax = cols * spacing + 0.8;
        const ymin = -0.8, ymax = rows * spacing + 0.8;

        const toSvgX = (x) => ((x - xmin) / (xmax - xmin) * width).toFixed(2);
        const toSvgY = (y) => ((y - ymin) / (ymax - ymin) * height).toFixed(2);

        let svg = `<?xml version="1.0" encoding="UTF-8"?>\n`;
        svg += `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">\n`;
        svg += `  <rect width="100%" height="100%" fill="#111115"/>\n`;
        svg += `  <g id="cells" stroke="#27ae60" stroke-width="1.8" fill="none" opacity="0.85">\n`;

        for (const poly of polys) {
            const pts = poly.map(p => `${toSvgX(p[0])},${toSvgY(p[1])}`).join(' ');
            svg += `    <polygon points="${pts}"/>\n`;
        }

        svg += `  </g>\n`;
        svg += `</svg>\n`;
        return svg;
    }
}

window.MoroniCore = MoroniCore;
