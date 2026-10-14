/**
 * MORONI WebGL Shader Engine
 * Real-time GPU implementation of MORONI: Cellular Texture and Fracture Synthesis via Roots-of-Unity Polygons and Signed Boundary Fields
 * Renders the 3-term energy function in parallel at 60+ FPS on consumer GPUs:
 *   E_c(x) = alpha * d_c + beta * s_c(x) + gamma * (1 - cos(kappa*theta + lambda*d_c))
 */

const VERTEX_SHADER_SRC = `#version 300 es
in vec2 a_position;
out vec2 v_uv;

void main() {
    v_uv = (a_position + 1.0) * 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER_SRC = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform vec2 u_resolution;
uniform float u_time;
uniform int u_mode;          // 0: Shaded/Color, 1: Normal Map, 2: Height Map, 3: Grout/Roughness, 4: dc, 5: sc, 6: phase, 7: split comparison
uniform float u_split_pos;   // 0.0 to 1.0 for A/B split comparison

// Grid & Cell parameters
uniform int u_rows;
uniform int u_cols;
uniform float u_spacing;
uniform float u_radius;
uniform float u_jitter;
uniform int u_sides;         // 3, 4, 5, 6, 8, or 0 (varied)
uniform float u_seed;

// Energy weights
uniform float u_alpha;       // Center distance weight
uniform float u_beta;        // Polygon SDF weight
uniform float u_gamma;       // Fracture phase weight
uniform float u_kappa;       // Phase angular harmonic (typically 2.0 or 3.0)
uniform float u_lambda;      // Phase radial frequency (typically 1.0)
uniform float u_margin;      // Grout / Mortar threshold
uniform bool u_use_margin;   // Toggle grout release vs full space-filling partition

// Visual styling
uniform vec3 u_grout_color;
uniform float u_edge_dark;
uniform bool u_show_wireframe;

#define PI 3.14159265358979323846
#define MAX_SIDES 12

// Hash functions for pseudo-random generation on GPU
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031 + u_seed);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973) + u_seed);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Signed distance to regular/jittered polygon
// Negative inside, positive outside
float sdPolygon(vec2 p, int n, float r, vec2 cellCoord) {
    float dmin = 1e6;
    float insideSign = 1.0;
    
    // Check inside via winding/cross product
    // For star-convex jittered roots of unity:
    float angle = atan(p.y, p.x);
    if (angle < 0.0) angle += 2.0 * PI;
    
    // Evaluate line segments of the jittered root polygon
    vec2 prevV = vec2(0.0);
    vec2 firstV = vec2(0.0);
    
    float fn = float(n);
    bool inside = true;
    
    for (int i = 0; i < MAX_SIDES; i++) {
        if (i >= n) break;
        float fi = float(i);
        float th = 2.0 * PI * fi / fn;
        // Angular jitter
        float jit = (hash12(cellCoord + vec2(fi * 1.31, fi * 0.77)) - 0.5) * 2.0 * u_jitter;
        th += jit;
        
        // Slight radial perturbation
        float r_jit = r * (0.92 + 0.16 * hash12(cellCoord + vec2(fi * 0.53, fi * 2.11)));
        vec2 currV = r_jit * vec2(cos(th), sin(th));
        
        if (i == 0) {
            firstV = currV;
        } else {
            // Distance from p to segment (prevV -> currV)
            vec2 e = currV - prevV;
            vec2 w = p - prevV;
            float t = clamp(dot(w, e) / (dot(e, e) + 1e-8), 0.0, 1.0);
            float dist = length(w - e * t);
            dmin = min(dmin, dist);
            
            // Cross product check for convexity orientation
            if ((e.x * w.y - e.y * w.x) < 0.0) {
                inside = false;
            }
        }
        prevV = currV;
    }
    
    // Final closing segment
    vec2 e = firstV - prevV;
    vec2 w = p - prevV;
    float t = clamp(dot(w, e) / (dot(e, e) + 1e-8), 0.0, 1.0);
    float dist = length(w - e * t);
    dmin = min(dmin, dist);
    if ((e.x * w.y - e.y * w.x) < 0.0) {
        inside = false;
    }
    
    return inside ? -dmin : dmin;
}

// Generate distinct procedural color per cell ID
vec3 cellColor(vec2 cellId) {
    float h = fract(sin(dot(cellId, vec2(12.9898, 78.233))) * 43758.5453);
    float s = 0.55 + 0.25 * fract(sin(dot(cellId, vec2(93.989, 67.345))) * 23145.21);
    float v = 0.80 + 0.18 * fract(sin(dot(cellId, vec2(45.123, 19.876))) * 87654.32);
    
    // HSV to RGB
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

// Algebraic phase interference for kappa=2
// 1 - cos(2*theta + lambda*d) = 1 - [(dx^2 - dy^2)*cos(lambda*d) - 2*dx*dy*sin(lambda*d)] / (d^2)
float phaseInterference(vec2 d, float dc, float kappa, float lam) {
    if (abs(kappa - 2.0) < 0.01) {
        float d2 = dot(d, d) + 1e-8;
        float cl = cos(lam * dc);
        float sl = sin(lam * dc);
        return 1.0 - ((d.x * d.x - d.y * d.y) * cl - 2.0 * d.x * d.y * sl) / d2;
    } else {
        float theta = atan(d.y, d.x);
        return 1.0 - cos(kappa * theta + lam * dc);
    }
}

// Evaluate MORONI energy at point p
struct EvalResult {
    vec2 bestCell;
    float bestEnergy;
    float dc;
    float sc;
    float phase;
    bool isGrout;
};

EvalResult evaluateField(vec2 p, bool forceVoronoi) {
    EvalResult res;
    res.bestEnergy = 1e8;
    res.bestCell = vec2(0.0);
    res.dc = 0.0;
    res.sc = 0.0;
    res.phase = 0.0;
    res.isGrout = false;
    
    vec2 baseGrid = floor(p / u_spacing);
    
    // 5x5 neighborhood search window guarantees admissible bound coverage
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            vec2 g = baseGrid + vec2(float(dx), float(dy));
            
            // Jittered cell center
            vec2 jitterOffset = (hash22(g) - 0.5) * 2.0 * (0.13 * u_spacing);
            vec2 center = (g + 0.5) * u_spacing + jitterOffset;
            
            vec2 delta = p - center;
            float dc = length(delta);
            
            float energy = 0.0;
            float sc = 0.0;
            float ph = 0.0;
            
            if (forceVoronoi) {
                // Classic Voronoi baseline
                energy = dc;
            } else {
                // Determine vertex count for this cell
                int sides = u_sides;
                if (sides == 0) {
                    // Varied between 5 and 8
                    sides = 5 + int(hash12(g * 7.13) * 4.0);
                }
                
                sc = sdPolygon(delta, sides, u_radius, g);
                ph = phaseInterference(delta, dc, u_kappa, u_lambda);
                
                energy = u_alpha * dc + u_beta * sc + u_gamma * ph;
            }
            
            if (energy < res.bestEnergy) {
                res.bestEnergy = energy;
                res.bestCell = g;
                res.dc = dc;
                res.sc = sc;
                res.phase = ph;
            }
        }
    }
    
    if (!forceVoronoi && u_use_margin && res.sc > u_margin) {
        res.isGrout = true;
    }
    
    return res;
}

void main() {
    // Transform UV to world coordinates
    float aspect = u_resolution.x / u_resolution.y;
    vec2 p = v_uv * vec2(float(u_cols), float(u_rows)) * u_spacing;
    
    bool forceVoronoi = false;
    if (u_mode == 7) {
        // Split comparison: Left is Voronoi, Right is MORONI
        if (v_uv.x < u_split_pos) {
            forceVoronoi = true;
        }
    }
    
    EvalResult ev = evaluateField(p, forceVoronoi);
    
    // Check if pixel is on the split line
    if (u_mode == 7 && abs(v_uv.x - u_split_pos) < 0.002) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    
    // Mode 4: Inspect dc (center distance)
    if (u_mode == 4) {
        float val = clamp(ev.dc / (u_spacing * 1.2), 0.0, 1.0);
        fragColor = vec4(vec3(val), 1.0);
        return;
    }
    
    // Mode 5: Inspect sc (polygon signed field)
    if (u_mode == 5) {
        // Red for outside, Blue for inside
        float norm_s = clamp(abs(ev.sc) / u_radius, 0.0, 1.0);
        vec3 col = (ev.sc < 0.0) ? mix(vec3(0.1, 0.3, 0.9), vec3(1.0), 1.0 - norm_s)
                                 : mix(vec3(0.9, 0.2, 0.1), vec3(1.0), 1.0 - norm_s);
        if (abs(ev.sc) < 0.015) col = vec3(0.0); // Zero-level set contour
        fragColor = vec4(col, 1.0);
        return;
    }
    
    // Mode 6: Inspect Phase Interference term
    if (u_mode == 6) {
        float val = clamp(ev.phase * 0.5, 0.0, 1.0);
        fragColor = vec4(vec3(val), 1.0);
        return;
    }
    
    // Mode 2: Height / Displacement Map
    if (u_mode == 2) {
        float h = 0.0;
        if (ev.isGrout) {
            h = 0.0;
        } else {
            // Convex height profile based on negative distance inside
            h = clamp(-ev.sc / u_radius, 0.0, 1.0);
            h = sqrt(h); // Dome profile
        }
        fragColor = vec4(vec3(h), 1.0);
        return;
    }
    
    // Mode 3: Grout / Roughness Map
    if (u_mode == 3) {
        float g = ev.isGrout ? 1.0 : 0.15;
        fragColor = vec4(vec3(g), 1.0);
        return;
    }
    
    // Mode 1: Normal Map (Screen-space / analytical derivative approximation)
    if (u_mode == 1) {
        vec2 eps = vec2(0.01 * u_spacing, 0.0);
        EvalResult ev_rx = evaluateField(p + eps.xy, forceVoronoi);
        EvalResult ev_lx = evaluateField(p - eps.xy, forceVoronoi);
        EvalResult ev_uy = evaluateField(p + eps.yx, forceVoronoi);
        EvalResult ev_dy = evaluateField(p - eps.yx, forceVoronoi);
        
        float h_c = ev.isGrout ? 0.0 : clamp(-ev.sc / u_radius, 0.0, 1.0);
        float h_rx = ev_rx.isGrout ? 0.0 : clamp(-ev_rx.sc / u_radius, 0.0, 1.0);
        float h_lx = ev_lx.isGrout ? 0.0 : clamp(-ev_lx.sc / u_radius, 0.0, 1.0);
        float h_uy = ev_uy.isGrout ? 0.0 : clamp(-ev_uy.sc / u_radius, 0.0, 1.0);
        float h_dy = ev_dy.isGrout ? 0.0 : clamp(-ev_dy.sc / u_radius, 0.0, 1.0);
        
        float dx = (h_rx - h_lx) * 4.0;
        float dy = (h_uy - h_dy) * 4.0;
        vec3 normal = normalize(vec3(-dx, -dy, 1.0));
        
        // Encode normal [-1, 1] to [0, 1] RGB
        vec3 normalRgb = normal * 0.5 + 0.5;
        fragColor = vec4(normalRgb, 1.0);
        return;
    }
    
    // Mode 0 & Mode 7 (Shaded Texture Color)
    if (ev.isGrout) {
        fragColor = vec4(u_grout_color, 1.0);
    } else {
        vec3 col = cellColor(ev.bestCell);
        
        // Edge detection via screen-space gradient
        vec2 eps = vec2(0.015 * u_spacing, 0.0);
        EvalResult ev_rx = evaluateField(p + eps.xy, forceVoronoi);
        EvalResult ev_uy = evaluateField(p + eps.yx, forceVoronoi);
        
        if (ev_rx.bestCell != ev.bestCell || ev_uy.bestCell != ev.bestCell ||
            ev_rx.isGrout || ev_uy.isGrout) {
            col *= u_edge_dark;
        }
        
        // Subtle height shading
        float h = clamp(-ev.sc / u_radius, 0.0, 1.0);
        col *= (0.75 + 0.35 * sqrt(h));
        
        fragColor = vec4(col, 1.0);
    }
}
`;

class MoroniGLRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true });
        if (!this.gl) {
            console.error('WebGL2 not supported, fallback needed.');
            this.gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });
        }
        this.program = null;
        this.quadVao = null;
        this.uniforms = {};
        this.initGL();
    }

    initGL() {
        const gl = this.gl;
        if (!gl) return;

        const vs = this.compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
        const fs = this.compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);

        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            console.error('Shader link error:', gl.getProgramInfoLog(this.program));
            return;
        }

        // Fullscreen quad
        const positions = new Float32Array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0
        ]);

        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

        const posAttr = gl.getAttribLocation(this.program, 'a_position');
        gl.enableVertexAttribArray(posAttr);
        gl.vertexAttribPointer(posAttr, 2, gl.FLOAT, false, 0, 0);

        // Cache uniform locations
        const uniformNames = [
            'u_resolution', 'u_time', 'u_mode', 'u_split_pos',
            'u_rows', 'u_cols', 'u_spacing', 'u_radius', 'u_jitter', 'u_sides', 'u_seed',
            'u_alpha', 'u_beta', 'u_gamma', 'u_kappa', 'u_lambda', 'u_margin', 'u_use_margin',
            'u_grout_color', 'u_edge_dark', 'u_show_wireframe'
        ];
        for (const name of uniformNames) {
            this.uniforms[name] = gl.getUniformLocation(this.program, name);
        }
    }

    compileShader(type, src) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, src);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        }
        return shader;
    }

    render(params, time = 0.0) {
        const gl = this.gl;
        if (!gl || !this.program) return;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.useProgram(this.program);

        gl.uniform2f(this.uniforms['u_resolution'], this.canvas.width, this.canvas.height);
        gl.uniform1f(this.uniforms['u_time'], time);
        gl.uniform1i(this.uniforms['u_mode'], params.mode || 0);
        gl.uniform1f(this.uniforms['u_split_pos'], params.splitPos !== undefined ? params.splitPos : 0.5);

        gl.uniform1i(this.uniforms['u_rows'], params.rows || 8);
        gl.uniform1i(this.uniforms['u_cols'], params.cols || 12);
        gl.uniform1f(this.uniforms['u_spacing'], params.spacing || 0.9);
        gl.uniform1f(this.uniforms['u_radius'], params.radius || 0.55);
        gl.uniform1f(this.uniforms['u_jitter'], params.jitter || 0.2);
        gl.uniform1i(this.uniforms['u_sides'], params.sides || 0);
        gl.uniform1f(this.uniforms['u_seed'], params.seed || 22.0);

        gl.uniform1f(this.uniforms['u_alpha'], params.alpha || 0.58);
        gl.uniform1f(this.uniforms['u_beta'], params.beta !== undefined ? params.beta : 0.32);
        gl.uniform1f(this.uniforms['u_gamma'], params.gamma !== undefined ? params.gamma : 0.10);
        gl.uniform1f(this.uniforms['u_kappa'], params.kappa || 2.0);
        gl.uniform1f(this.uniforms['u_lambda'], params.lambda || 1.0);
        gl.uniform1f(this.uniforms['u_margin'], params.margin !== undefined ? params.margin : 0.03);
        gl.uniform1i(this.uniforms['u_use_margin'], params.useMargin ? 1 : 0);

        const gc = params.groutColor || [0.07, 0.07, 0.07];
        gl.uniform3f(this.uniforms['u_grout_color'], gc[0], gc[1], gc[2]);
        gl.uniform1f(this.uniforms['u_edge_dark'], params.edgeDark || 0.27);
        gl.uniform1i(this.uniforms['u_show_wireframe'], params.showWireframe ? 1 : 0);

        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
}

window.MoroniGLRenderer = MoroniGLRenderer;
