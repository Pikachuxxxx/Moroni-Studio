/**
 * MORONI Core JavaScript Engine
 * Mathematical implementation of MORONI, Voronoi, Power Diagrams, Apollonius, and Worley noise.
 * Includes descriptive raster metrics and illustrative material-map exporters.
 */

class MoroniCore {
    static cellColors(n,seed=33) {
        const draw=this.createPRNG(seed);
        const hues=Array.from({length:n},(_,i)=>(i/n+.04*(draw()-.5)+1)%1);
        const sats=Array.from({length:n},()=>.46+.24*draw());
        const vals=Array.from({length:n},()=>.68+.22*draw());
        return hues.map((h,i)=>this.hsvToRgb(h,sats[i],vals[i]).map(v=>v/255));
    }
    // Generate jittered roots of unity polygon
    static generateRootPolygon(center, n, r, jitter, prng) {
        const poly = [];
        const j = Math.min(jitter, 0.45 * Math.PI / n);
        for (let k = 0; k < n; k++) {
            const th = 2 * Math.PI * k / n + (prng() * 2 - 1) * j;
            poly.push([center[0] + r * Math.cos(th), center[1] + r * Math.sin(th)]);
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
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;

            // Distance to segment [poly[j], poly[i]]
            const vx = xi - xj, vy = yi - yj;
            const wx = x - xj, wy = y - yj;
            const vsq = Math.max(vx * vx + vy * vy, 1e-300);
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
        const {rows=8, cols=12, spacing=.9, radius=.5, jitter=.35, sides=0, seed=22} = params;
        if (![rows,cols,spacing,radius,jitter,seed,sides].every(Number.isFinite) ||
            !Number.isInteger(rows) || !Number.isInteger(cols) || !Number.isInteger(sides) ||
            rows*cols > 256 || rows<1 || cols<1 || spacing<=0 || radius<=0 || jitter<0)
            throw new Error('Supported field: 1..256 cells, positive spacing/radius');
        const prng = MoroniCore.createPRNG(seed);
        const centers=[], polys=[], weights=[], radii=[];
        for (let r=0;r<rows;r++) for(let c=0;c<cols;c++) {
            const center=[(c+.26*(prng()-.5))*spacing,(r+.26*(prng()-.5))*spacing];
            const n=(sides || 5)+Math.floor(prng()*(sides ? 1 : 4));
            if(n<3 || n>12) throw new Error('Sides must be 3..12');
            const rad=radius*(.9+.2*prng());
            centers.push(center); radii.push(rad); weights.push(rad*rad);
            polys.push(MoroniCore.generateRootPolygon(center,n,rad,jitter,prng));
        }
        return {rows,cols,spacing,centers,polys,weights,radii,
            bbox:[-spacing/2,(cols-.5)*spacing,-spacing/2,(rows-.5)*spacing]};
    }

    // Render CPU image buffer with requested generator
    static renderCPU(field,width,height,params) {
        const {alpha=.58,beta=.32,gamma=.1,kappa=2,margin=0,useMargin=true,generator='moroni'}=params;
        const lam=params.lam ?? params.lambda ?? 1;
        if (![alpha,beta,gamma,lam].every(v=>Number.isFinite(v)&&v>=0) ||
            !Number.isInteger(kappa) || kappa<1 || kappa>6 || !Number.isFinite(margin) ||
            !Number.isInteger(width) || !Number.isInteger(height) || width<1 || height<1)
            throw new Error('Invalid energy parameters or image dimensions');
        const {centers,polys,weights,bbox}=field;
        const [xmin,xmax,ymin,ymax]=bbox;
        const labels=new Int32Array(width*height), distances=new Float64Array(width*height),
              sdfs=new Float64Array(width*height), gaps=new Float64Array(width*height);
        for(let py=0;py<height;py++) for(let px=0;px<width;px++) {
            const x=xmin+(px+.5)*(xmax-xmin)/width,y=ymin+(py+.5)*(ymax-ymin)/height;
            let best=Infinity,second=Infinity,id=0,scBest=0,dcBest=0;
            for(let i=0;i<centers.length;i++) {
                const dx=x-centers[i][0],dy=y-centers[i][1],d=Math.hypot(dx,dy);
                const sc=MoroniCore.pointPolygonSignedDistance(x,y,polys[i]);
                const ph=d<1e-6 ? 0 : 1-Math.cos(kappa*Math.atan2(dy,dx)+lam*d);
                let e=alpha*d+beta*sc+gamma*ph;
                if(generator==='voronoi_l2') e=d;
                if(generator==='power') e=d*d-weights[i];
                if(generator==='apollonius') e=d-Math.sqrt(weights[i]);
                if(e<best) {second=best;best=e;id=i;scBest=sc;dcBest=d;}
                else second=Math.min(second,e);
            }
            const j=py*width+px;
            labels[j]=useMargin && scBest>margin ? -1 : id;
            distances[j]=dcBest;sdfs[j]=scBest;gaps[j]=second-best;
        }
        return {labels,distances,sdfs,gaps,width,height,bbox,nCells:centers.length,spacing:field.spacing};
    }

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
            if (pts.length < Math.max(1,Math.ceil(width*height*.0001))) continue;
            validCells++;

            // Exact hull of the sampled pixel corners, matching the CPU metric.
            const corners=pts.flatMap(([x,y])=>[[x-.5,y-.5],[x-.5,y+.5],[x+.5,y-.5],[x+.5,y+.5]]);
            const chArea = MoroniCore.convexHullArea(corners);
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
        const middle = Math.floor(solidities.length / 2);
        const medianSolidity = !solidities.length ? 1.0 : solidities.length % 2 ?
            solidities[middle] : (solidities[middle - 1] + solidities[middle]) / 2;

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

        // Dropping corner samples can shrink the hull and make solidity exceed 1.
        const sample = points.slice();

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
        return Math.abs(area)*0.5;
    }

    // Count connected components via 8-connected BFS
    static countConnectedComponents(points, width, height) {
        const pointSet = new Set(points.map(p => `${p[0]},${p[1]}`));
        let components = 0;

        for (const p of points) {
            const key = `${p[0]},${p[1]}`;
            if (!pointSet.has(key)) continue;

            let size=0;
            const queue = [p];
            pointSet.delete(key);

            while (queue.length > 0) {
                const [cx, cy] = queue.pop(); size++;
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
            if(size>=Math.max(1,Math.ceil(width*height*.0001))) components++;
        }
        return components;
    }

    // Fallback maps: shared unlit palette and height; normals use raster differences.
    // GPU normals evaluate at fixed world offsets instead, so fallback normals differ.
    static generatePBRMaps(result, radius) {
        const {labels,sdfs,width,height,bbox,nCells,spacing}=result;
        const colors=this.cellColors(nCells), count=width*height;
        const maps=Object.fromEntries(['diffuse','normal','height','roughness'].map(k=>[k,new Uint8ClampedArray(4*count)]));
        const heights=Float64Array.from(sdfs,(s,i)=>labels[i]<0?0:Math.sqrt(Math.max(0,Math.min(1,-s/radius))));
        const dx=(bbox[1]-bbox[0])/width,dy=(bbox[3]-bbox[2])/height;
        for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
            const i=y*width+x, left=Math.max(0,x-1),right=Math.min(width-1,x+1),
                  top=Math.max(0,y-1),bottom=Math.min(height-1,y+1);
            const gx=(heights[y*width+right]-heights[y*width+left])/(Math.max(1,right-left)*dx),
                  gy=(heights[bottom*width+x]-heights[top*width+x])/(Math.max(1,bottom-top)*dy);
            const nx=-.06*spacing*gx,ny=-.06*spacing*gy,len=Math.hypot(nx,ny,1);
            const color=labels[i]<0?[.07,.07,.07]:colors[labels[i]],rough=labels[i]<0?.95:.25;
            for(let c=0;c<3;c++) {
                maps.diffuse[4*i+c]=Math.round(255*color[c]);
                maps.height[4*i+c]=Math.round(255*heights[i]);
                maps.roughness[4*i+c]=Math.round(255*rough);
                maps.normal[4*i+c]=Math.round(255*(.5+.5*[nx,ny,1][c]/len));
            }
            for(const data of Object.values(maps))data[4*i+3]=255;
        }
        return Object.fromEntries(Object.entries(maps).map(([k,data])=>[k,new ImageData(data,width,height)]));
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
        return [r * 255, g * 255, b * 255];
    }

    // Export SVG vector boundaries and polygons
    static exportSVG(field, width, height) {
        const { cols, rows, spacing, polys } = field;
        const [xmin,xmax,ymin,ymax]=field.bbox;


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

if(typeof window!=='undefined') window.MoroniCore=MoroniCore;
if(typeof module!=='undefined') module.exports=MoroniCore;
