# Reproducing the revised artifact

## Environment and tests

The recorded CPU run used Windows, AMD Ryzen 9 7900X, Python 3.12, NumPy 2.3.5, SciPy 1.18.1 and Matplotlib 3.11.2. JSON records exact interpreter/library versions and the SHA-256 of `src/moroni.py`. GPU results identify the user agent and AMD Radeon RX 9070 via ANGLE/D3D11. Timing varies with hardware, driver and other workload.

Create/activate a Python environment and install `requirements.txt`. To match the numerical run, use `requirements-reproduction.txt`. Then run from the repository root:

```text
python -m unittest discover -s tests -v
node tests/test_core.js
python benchmarks/reproduce.py all
```

The full run can take several minutes, especially full-search CPU timings. Commands `verify`, `control`, `bench` and `figures` run individual groups. `control` includes LUT and raster experiments. `figures` refreshes the five CPU tables and figures from the recorded experiment data; it also computes the extra unclipped shape, coverage and raster rows. To refresh tables alone, run `python benchmarks/paper_tables.py`. `experiments.py` and `lut_and_baselines_eval.py` are compatibility entry points for the current pipeline. Historical prototype scripts are not required.

The pipeline regenerates correctness/control/LUT/raster/CPU timing JSON, the cross-language fixture, the five CPU data tables directly inside `paper/paper.tex`, and the original figure filenames and panel layouts. Settings and all measured samples are retained. Search tests compare each distance mode to its corresponding full search; analytical-versus-LUT error is measured separately. A target's eligibility for IoU is independent of retained output, so vanished outputs score zero.

## Genuine browser measurements and figure

```text
python -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765/app/validation.html and select **Run validation and benchmark**. Download **JSON report** and **map figure**, then run:

```text
python benchmarks/import_browser.py /path/to/browser-validation.json /path/to/browser_maps.png
```

Use quoted absolute paths on Windows when paths contain spaces. Run and import before modifying the tested browser sources. The import records their hashes, replaces the existing GPU table inside `paper/paper.tex`, and updates `paper/figures/pbr_showcase.png`; it does not synthesize GPU results. The figure is four actual GPU renders captured by the browser. If timer queries are unavailable, the table explicitly says so.

Validation compares uploaded geometry/palette and the default labels with Python, then runs nine cases at 96 x 64. The automated gate requires zero GPU pruned/full differences and at most 0.1% CPU/GPU differences per case; the paper reports the actual observed mismatch count and energy lead, not that threshold as a guarantee. There are 12 warmups and 40 measured draws per timing condition. Disjoint intervals are discarded and counted. Readback, field upload, JS and PNG encoding are outside GPU timings. Raw samples permit independent summary statistics.

## Build the paper

The manuscript retains its original `acmart` two-column layout, figures, eight tables and five appendix sections. Tables are inline in `paper.tex`; `references.bib` contains the bibliography. From `paper/`, either:

```text
tectonic --keep-logs paper.tex
```

or run `pdflatex -interaction=nonstopmode -halt-on-error paper.tex`, `bibtex paper`, then `pdflatex` twice more with the same arguments. The delivered PDF was built with Tectonic 0.17.0 and visually checked after rendering. Standard LaTeX packages may be downloaded on first Tectonic use.

On a new experiment run, inspect changed results and update the manuscript's quoted numerical examples, environment and mismatch statements before publishing. The table refresh replaces only existing tabular blocks. It preserves surrounding captions and prose, which must be checked against a new run before publishing.

The studio screenshot is a genuine browser capture; refresh it manually after UI changes. The browser map figure must come from the validation-page download, not the CPU plotting script.

## Interpretation

The current evidence supports a practical finite texture recipe, conservative real-arithmetic bounds, sampled implementation agreement and modest silhouette arbitration improvement beyond matched clipping. It does not establish a new Voronoi family, perceptual superiority, physical fracture accuracy, universal GPU speed or acceptance by a venue. New claims require new evidence.
