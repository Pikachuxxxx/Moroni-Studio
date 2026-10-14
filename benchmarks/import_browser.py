"""Import genuine validation-page downloads and regenerate the GPU table.
Usage: python benchmarks/import_browser.py REPORT.json MAPS.png
"""
import argparse,json,shutil,statistics,hashlib
from pathlib import Path
from PIL import Image
from manuscript_tables import replace_table
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('report',type=Path);p.add_argument('figure',type=Path)
a=p.parse_args();report=json.loads(a.report.read_text())
if not report.get('passed'):raise SystemExit('Report failed validation; inspect it before publishing results.')
with Image.open(a.figure) as im:
    if im.size!=(2400,442):raise SystemExit('Expected the four-panel figure downloaded by validation.html')
report['imported_source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in ['app/moroni-core.js','app/moroni-glsl.js','app/validation.html','tests/field_fixture.json']}
report['import_note']='Source hashes captured at import; run validation and import before editing these sources.'
lines=[r'\begin{tabular}{r l r r}\toprule',r'Pixels & Output & Draws & Median ms \\ \midrule']
for row in report['timings']:
    if not row['raw_ms']:continue
    row['median_ms']=statistics.median(row['raw_ms'])
    mode={9:'Unlit color',1:'Normal'}.get(row['mode'],str(row['mode']))
    lines.append(f"{row['width']} $\\times$ {row['height']} & {mode} & {len(row['raw_ms'])} & {row['median_ms']:.3f} \\\\")
if len(lines)==2:lines.append(r'\multicolumn{4}{c}{Timer query unavailable on this browser/device} \\')
lines.append(r'\bottomrule\end{tabular}')
replace_table('tbl:gpu_perf','\n'.join(lines))
(ROOT/'benchmarks/results/browser-validation.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
destination=ROOT/'paper/figures/pbr_showcase.png'
if a.figure.resolve()!=destination.resolve():shutil.copyfile(a.figure,destination)
print('Imported validated browser measurements, GPU table, and map figure.')
