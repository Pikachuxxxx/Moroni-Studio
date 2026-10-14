/**
 * MORONI Studio â€” Unity-Style Application Controller
 * Manages 12 illustrative material presets, WebGL2 GPU engine with CPU 2D fallback,
 * live telemetry HUD, PBR map generation, and SVG vector export.
 */

// 12 Production Presets
const PRESETS = {
    stone_wall: {
        name: "Medieval Stone Wall",
        sides: 5,
        radius: 0.62,
        jitter: 0.12,
        margin: 0.03,
        useMargin: true,
        alpha: 0.58,
        beta: 0.90,
        gamma: 0.02,
        kappa: 2.0,
        lambda: 1.0,
        groutColor: [0.18, 0.18, 0.18],
    },
    cracked_mud: {
        name: "Cracked Mud / Desiccation",
        sides: 7,
        radius: 0.60,
        jitter: 0.40,
        margin: 0.05,
        useMargin: true,
        alpha: 0.58,
        beta: 0.32,
        gamma: 0.30,
        kappa: 2.0,
        lambda: 1.0,
        groutColor: [0.15, 0.12, 0.09],
    },
    ceramic_tile: {
        name: "Ceramic Floor Tile",
        sides: 4,
        radius: 0.60,
        jitter: 0.02,
        margin: 0.01,
        useMargin: true,
        alpha: 0.58,
        beta: 1.20,
        gamma: 0.00,
        kappa: 2.0,
        lambda: 1.0,
        groutColor: [0.90, 0.90, 0.92],
    },
    shattered_glass: {
        name: "Shattered Tempered Glass",
        sides: 4,
        radius: 0.70,
        jitter: 0.50,
        margin: 0.08,
        useMargin: true,
        alpha: 0.58,
        beta: 0.32,
        gamma: 0.35,
        kappa: 3.0,
        lambda: 1.8,
        groutColor: [0.03, 0.05, 0.08],
    },
    spiderweb: {
        name: "Spiderweb Radial Fracture",
        sides: 6,
        radius: 0.65,
        jitter: 0.30,
        margin: 0.04,
        useMargin: true,
        alpha: 0.50,
        beta: 0.35,
        gamma: 0.42,
        kappa: 4.0,
        lambda: 2.5,
        groutColor: [0.05, 0.05, 0.05],
    },
    cobblestone: {
        name: "Cobblestone Street Pavement",
        sides: 6,
        radius: 0.64,
        jitter: 0.25,
        margin: 0.06,
        useMargin: true,
        alpha: 0.55,
        beta: 0.75,
        gamma: 0.05,
        kappa: 2.0,
        lambda: 1.2,
        groutColor: [0.12, 0.12, 0.14],
    },
    scifi_hex: {
        name: "Sci-Fi Hexagonal Hull Plating",
        sides: 6,
        radius: 0.60,
        jitter: 0.00,
        margin: 0.02,
        useMargin: true,
        alpha: 0.58,
        beta: 1.35,
        gamma: 0.00,
        kappa: 2.0,
        lambda: 1.0,
        groutColor: [0.05, 0.15, 0.25],
    },
    craquelure: {
        name: "Porcelain Craquelure Glaze",
        sides: 5,
        radius: 0.55,
        jitter: 0.35,
        margin: 0.015,
        useMargin: true,
        alpha: 0.60,
        beta: 0.40,
        gamma: 0.22,
        kappa: 2.0,
        lambda: 2.2,
        groutColor: [0.35, 0.25, 0.15],
    },
    dragon_scales: {
        name: "Dragon Scales / Reptilian",
        sides: 5,
        radius: 0.68,
        jitter: 0.18,
        margin: 0.02,
        useMargin: true,
        alpha: 0.52,
        beta: 0.85,
        gamma: 0.08,
        kappa: 1.0,
        lambda: 1.4,
        groutColor: [0.08, 0.12, 0.05],
    },
    alien_biomass: {
        name: "Alien Biomass Tissue",
        sides: 6,
        radius: 0.62,
        jitter: 0.45,
        margin: 0.04,
        useMargin: true,
        alpha: 0.48,
        beta: 0.45,
        gamma: 0.28,
        kappa: 3.0,
        lambda: 1.5,
        groutColor: [0.22, 0.05, 0.18],
    },
    basalt_columns: {
        name: "Basalt Prisms (Giant's Causeway)",
        sides: 6,
        radius: 0.60,
        jitter: 0.08,
        margin: 0.012,
        useMargin: true,
        alpha: 0.58,
        beta: 1.10,
        gamma: 0.00,
        kappa: 2.0,
        lambda: 1.0,
        groutColor: [0.10, 0.10, 0.10],
    },
    voronoi_baseline: {
        name: "Classic Voronoi Baseline",
        sides: 6,
        radius: 0.55,
        jitter: 0.00,
        margin: 0.00,
        useMargin: false,
        alpha: 1.00,
        beta: 0.00,
        gamma: 0.00,
        kappa: 2.0,
        lambda: 1.0,
        groutColor: [0.05, 0.05, 0.05],
    }
};

class MoroniApp {
    constructor() {
        this.canvas = document.getElementById('viewport');
        this.useWebGL = true;
        
        // Try WebGL renderer
        try {
            this.glRenderer = new MoroniGLRenderer(this.canvas);
            if (!this.glRenderer.gl || !this.glRenderer.program) {
                this.useWebGL = false;
            }
        } catch (e) {
            console.warn("WebGL initialization failed, falling back to 2D canvas:", e);
            this.useWebGL = false;
        }

        if (!this.useWebGL) {
            const replacement=this.canvas.cloneNode();this.canvas.replaceWith(replacement);this.canvas=replacement;
            this.ctx2d = this.canvas.getContext('2d');
        }

        this.state = {
            rows: 8,
            cols: 12,
            spacing: 0.9,
            radius: 0.62,
            jitter: 0.12,
            sides: 5,
            seed: 22.0,
            
            alpha: 0.58,
            beta: 0.90,
            gamma: 0.02,
            kappa: 2.0,
            lambda: 1.0,
            margin: 0.03,
            useMargin: true,
            
            groutColor: [0.18, 0.18, 0.18],
            showWireframe: false,
            
            mode: 9, // Unlit paper colors.
            splitPos: 0.5,
            isDraggingSplit: false,
            isPlaying: true
        };

        this.fpsCounter = document.getElementById('fps-val');
        this.statGpu = document.getElementById('stat-gpu');
        this.lastTime = performance.now();
        this.frameCount = 0;
        this.fps = 60;
        this.metricsDebounce = null;

        if (!this.useWebGL) {
            document.querySelectorAll('.view-btn').forEach(b=>{if(![9,1,2,3].includes(Number(b.dataset.mode)))b.disabled=true;});
            if(this.statGpu)this.statGpu.textContent='Unavailable (CPU fallback)';
        }
        this.bindEvents();
        this.loadPreset('stone_wall');
        this.resizeCanvas();
        this.forceRenderFrame(); // Render immediately so canvas is never blank!
        this.startLoop();
    }

    bindEvents() {
        // Slider bindings
        const bindSlider = (id, stateKey, parser = parseFloat, isInt = false) => {
            const slider = document.getElementById(id);
            const num = document.getElementById(`${id}-val`);
            if (!slider) return;

            const update = (val) => {
                const parsed = parser(val);
                this.state[stateKey] = parsed;
                slider.value = parsed;
                if (num) num.textContent = isInt ? parsed : parsed.toFixed(2);
                this.forceRenderFrame();
                this.scheduleMetricsUpdate();
            };

            slider.addEventListener('input', (e) => update(e.target.value));
        };

        bindSlider('param-sides', 'sides', parseInt, true);
        bindSlider('param-radius', 'radius');
        bindSlider('param-jitter', 'jitter');
        bindSlider('param-margin', 'margin');
        bindSlider('param-gamma', 'gamma');
        bindSlider('param-beta', 'beta');
        bindSlider('param-alpha', 'alpha');
        bindSlider('param-kappa', 'kappa', parseFloat);
        bindSlider('param-lambda', 'lambda', parseFloat);

        // Preset Dropdown
        const presetSelect = document.getElementById('preset-select');
        if (presetSelect) {
            presetSelect.addEventListener('change', (e) => {
                this.loadPreset(e.target.value);
            });
        }

        // Margin Toggle
        const marginToggle = document.getElementById('param-use-margin');
        if (marginToggle) {
            marginToggle.addEventListener('change', (e) => {
                this.state.useMargin = e.target.checked;
                this.forceRenderFrame();
                this.scheduleMetricsUpdate();
            });
        }

        // Random Seed Button
        const seedBtn = document.getElementById('btn-randomize-seed');
        if (seedBtn) {
            seedBtn.addEventListener('click', () => {
                this.state.seed = Math.floor(Math.random() * 10000);
                document.getElementById('param-seed-val').textContent = this.state.seed;
                this.forceRenderFrame();
                this.scheduleMetricsUpdate();
            });
        }

        // Wireframe Toggle
        const wireToggle = document.getElementById('toggle-wireframe');
        if (wireToggle) {
            wireToggle.addEventListener('change', (e) => {
                this.state.showWireframe = e.target.checked;
                this.forceRenderFrame();
            });
        }

        // Project Asset Cards
        const assetCards = document.querySelectorAll('.asset-card');
        assetCards.forEach(card => {
            card.addEventListener('click', () => {
                assetCards.forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                const presetKey = card.dataset.preset;
                if (presetKey) this.loadPreset(presetKey);
            });
        });

        // View Mode Buttons
        const modeButtons = document.querySelectorAll('.view-btn');
        modeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                modeButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.state.mode = parseInt(btn.dataset.mode);
                const splitHint = document.getElementById('split-hint');
                if (splitHint) {
                    splitHint.style.display = (this.state.mode === 7) ? 'block' : 'none';
                }
                this.forceRenderFrame();
            });
        });

        // Split Comparison Slider Dragging
        const handleSplitMove = (clientX) => {
            if (this.state.mode !== 7) return;
            const rect = this.canvas.getBoundingClientRect();
            let normX = (clientX - rect.left) / rect.width;
            normX = Math.max(0.01, Math.min(0.99, normX));
            this.state.splitPos = normX;
            this.forceRenderFrame();
        };

        this.canvas.addEventListener('mousedown', (e) => {
            if (this.state.mode === 7) {
                this.state.isDraggingSplit = true;
                handleSplitMove(e.clientX);
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (this.state.isDraggingSplit) {
                handleSplitMove(e.clientX);
            }
        });

        window.addEventListener('mouseup', () => {
            this.state.isDraggingSplit = false;
        });

        // Play/Pause
        document.getElementById('btn-play')?.addEventListener('click', () => {
            this.state.isPlaying = true;
            document.getElementById('btn-play').classList.add('playing');
        });
        document.getElementById('btn-pause')?.addEventListener('click', () => {
            this.state.isPlaying = false;
            document.getElementById('btn-play').classList.remove('playing');
        });

        document.getElementById('sample-four')?.addEventListener('change',e=>{
            this.state.samples=e.target.checked ? 4:1;this.forceRenderFrame();
        });
        // Window resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.forceRenderFrame();
        });

        // Export Buttons
        document.getElementById('btn-export-diffuse')?.addEventListener('click', () => this.exportMap('diffuse'));
        document.getElementById('btn-export-normal')?.addEventListener('click', () => this.exportMap('normal'));
        document.getElementById('btn-export-height')?.addEventListener('click', () => this.exportMap('height'));
        document.getElementById('btn-export-roughness')?.addEventListener('click', () => this.exportMap('roughness'));
        document.getElementById('btn-export-svg')?.addEventListener('click', () => this.exportSVG());
        document.getElementById('btn-export-all')?.addEventListener('click', () => this.exportAllPBR());
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        if (!container) return;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const aspect=this.state.cols/this.state.rows;
        const width = Math.min(container.clientWidth || 800,(container.clientHeight || 500)*aspect);
        const height = width/aspect;
        this.canvas.style.width=width+'px';
        this.canvas.style.height=height+'px';

        if (this.canvas.width !== width * dpr || this.canvas.height !== height * dpr) {
            this.canvas.width = width * dpr;
            this.canvas.height = height * dpr;
        }
    }

    loadPreset(presetKey) {
        const p = PRESETS[presetKey];
        if (!p) return;

        Object.assign(this.state, p);

        const setVal = (id, val, isInt = false) => {
            const slider = document.getElementById(id);
            const num = document.getElementById(`${id}-val`);
            if (slider) slider.value = val;
            if (num) num.textContent = isInt ? val : val.toFixed(2);
        };

        setVal('param-sides', p.sides, true);
        setVal('param-radius', p.radius);
        setVal('param-jitter', p.jitter);
        setVal('param-margin', p.margin);
        setVal('param-gamma', p.gamma);
        setVal('param-beta', p.beta);
        setVal('param-alpha', p.alpha);
        setVal('param-kappa', p.kappa);
        setVal('param-lambda', p.lambda);

        const marginToggle = document.getElementById('param-use-margin');
        if (marginToggle) marginToggle.checked = p.useMargin;

        const presetSelect = document.getElementById('preset-select');
        if (presetSelect && presetSelect.value !== presetKey) {
            presetSelect.value = presetKey;
        }

        this.forceRenderFrame();
        this.scheduleMetricsUpdate();
    }

    scheduleMetricsUpdate() {
        if (this.metricsDebounce) clearTimeout(this.metricsDebounce);
        this.metricsDebounce = setTimeout(() => this.updateTopologyMetrics(), 200);
    }

    updateTopologyMetrics() {
        const width=180,height=120;
        let result;
        if(this.useWebGL && this.glRenderer) {
            result=this.glRenderer.readLabels(this.state,width,height);
            this.forceRenderFrame();
        } else {
            result=MoroniCore.renderCPU(MoroniCore.buildCellField(this.state),width,height,this.state);
        }
        const metrics=MoroniCore.computeMetrics(result.labels,width,height,this.state.rows*this.state.cols);
        document.getElementById('metric-solidity').textContent=metrics.medianSolidity;
        document.getElementById('metric-nonconvex').textContent=metrics.nonConvexPct+'%';
        document.getElementById('metric-disconnected').textContent=metrics.disconnectedPct+'%';
        document.getElementById('falsification-badge').textContent='Raster sample: 180 x 120, 8-connected';
    }

    forceRenderFrame(time = 0.0) {
        if (this.useWebGL && this.glRenderer && this.glRenderer.gl) {
            this.glRenderer.render(this.state, time);
            if(this.statGpu) this.statGpu.textContent=this.glRenderer.gpuMs===null ?
                (this.glRenderer.timer ? 'Pending' : 'Unavailable') : this.glRenderer.gpuMs.toFixed(2)+' ms';
        } else if (this.ctx2d) {
            // High-quality CPU 2D render fallback
            const w = this.canvas.width;
            const h = this.canvas.height;
            const field = MoroniCore.buildCellField({
                rows: this.state.rows,
                cols: this.state.cols,
                spacing: this.state.spacing,
                radius: this.state.radius,
                jitter: this.state.jitter,
                sides: this.state.sides,
                seed: this.state.seed
            });
            const renderRes = MoroniCore.renderCPU(field, w, h, {
                generator: (this.state.mode === 7 && this.state.splitPos > 0.5) ? 'voronoi_l2' : 'moroni',
                alpha: this.state.alpha,
                beta: this.state.beta,
                gamma: this.state.gamma,
                kappa: this.state.kappa,
                lam: this.state.lambda,
                margin: this.state.margin,
                useMargin: this.state.useMargin
            });
            const pbr = MoroniCore.generatePBRMaps(renderRes, this.state.radius);
            this.ctx2d.putImageData(pbr[({1:'normal',2:'height',3:'roughness'})[this.state.mode]] || pbr.diffuse, 0, 0);
        }
    }

    startLoop() {
        const render = (time) => {
            this.frameCount++;
            const now = performance.now();
            if (now - this.lastTime >= 1000) {
                this.fps = Math.round((this.frameCount * 1000) / (now - this.lastTime));
                if (this.fpsCounter) this.fpsCounter.textContent = `${this.fps}.0 FPS (${(1000/Math.max(1,this.fps)).toFixed(1)}ms)`;
                this.frameCount = 0;
                this.lastTime = now;
            }

            if (this.state.isPlaying && this.useWebGL) {
                this.forceRenderFrame(time * 0.001);
            }
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    getExportResolution() {
        const resSelect = document.getElementById('export-res-select');
        const dim = resSelect ? parseInt(resSelect.value) : 1024;
        return { width: dim, height: Math.round(dim * (this.state.rows / this.state.cols)) };
    }

    exportMap(mapType) {
        const { width, height } = this.getExportResolution();
        const exportStatus = document.getElementById('export-status');
        if (exportStatus) exportStatus.textContent = `Baking ${mapType.toUpperCase()} map (${width}Ã—${height})...`;

        setTimeout(() => {
            if (this.useWebGL && this.glRenderer && this.glRenderer.gl) {
                // Render directly using GPU WebGL fragment shader
                const origW = this.canvas.width;
                const origH = this.canvas.height;
                this.canvas.width = width;
                this.canvas.height = height;
                this.glRenderer.gl.viewport(0, 0, width, height);

                const modeMap = { diffuse: 9, normal: 1, height: 2, roughness: 3 };
                const exportMode = modeMap[mapType] !== undefined ? modeMap[mapType] : 0;
                const exportState = { ...this.state, mode: exportMode, samples: 1, showWireframe: false };

                this.glRenderer.render(exportState, 0.0);

                // Snapshot before restoring the display; use a Blob URL for downloads.
                const snapshot = document.createElement('canvas');
                snapshot.width=width; snapshot.height=height;
                snapshot.getContext('2d').drawImage(this.canvas,0,0);
                this.downloadPNG(snapshot, `moroni_${mapType}_${width}x${height}.png`);

                // Restore display canvas resolution
                this.canvas.width = origW;
                this.canvas.height = origH;
                this.glRenderer.gl.viewport(0, 0, origW, origH);
                this.forceRenderFrame();

                if (exportStatus) exportStatus.textContent = `Successfully exported GPU ${mapType.toUpperCase()}!`;
                setTimeout(() => { if (exportStatus) exportStatus.textContent = ""; }, 3000);
            } else {
                const field = MoroniCore.buildCellField({
                    rows: this.state.rows,
                    cols: this.state.cols,
                    spacing: this.state.spacing,
                    radius: this.state.radius,
                    jitter: this.state.jitter,
                    sides: this.state.sides,
                    seed: this.state.seed
                });

                const renderRes = MoroniCore.renderCPU(field, width, height, {
                    generator: 'moroni',
                    alpha: this.state.alpha,
                    beta: this.state.beta,
                    gamma: this.state.gamma,
                    kappa: this.state.kappa,
                    lam: this.state.lambda,
                    margin: this.state.margin,
                    useMargin: this.state.useMargin
                });

                const pbr = MoroniCore.generatePBRMaps(renderRes, this.state.radius);
                const imgData = pbr[mapType];

                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = width;
                tempCanvas.height = height;
                const ctx = tempCanvas.getContext('2d');
                ctx.putImageData(imgData, 0, 0);

                this.downloadPNG(tempCanvas, `moroni_${mapType}_${width}x${height}.png`);

                if (exportStatus) exportStatus.textContent = `Successfully baked ${mapType.toUpperCase()}!`;
                setTimeout(() => { if (exportStatus) exportStatus.textContent = ""; }, 3000);
            }
        }, 30);
    }

    downloadPNG(canvas, filename) {
        canvas.toBlob(blob => {
            if (!blob) return;
            const link=document.createElement('a'), url=URL.createObjectURL(blob);
            link.href=url; link.download=filename; link.click();
            setTimeout(()=>URL.revokeObjectURL(url),1000);
        }, 'image/png');
    }

    exportSVG() {
        const { width, height } = this.getExportResolution();
        const field = MoroniCore.buildCellField({
            rows: this.state.rows,
            cols: this.state.cols,
            spacing: this.state.spacing,
            radius: this.state.radius,
            jitter: this.state.jitter,
            sides: this.state.sides,
            seed: this.state.seed
        });

        const svgContent = MoroniCore.exportSVG(field, width, height);
        const blob = new Blob([svgContent], { type: 'image/svg+xml' });
        const link = document.createElement('a');
        link.download = `moroni_input_polygons_${width}x${height}.svg`;
        link.href = URL.createObjectURL(blob);
        link.click();
        setTimeout(()=>URL.revokeObjectURL(link.href),1000);
    }

    exportAllPBR() {
        ['diffuse', 'normal', 'height', 'roughness'].forEach((type, idx) => {
            setTimeout(() => this.exportMap(type), idx * 400);
        });
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.app = new MoroniApp();
});
