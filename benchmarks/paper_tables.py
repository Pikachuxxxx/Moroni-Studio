"""Regenerate data for the manuscript's original table layout; preserve figures.
Run after the CPU experiments. No manuscript prose or image assets are modified.
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import label,generate_binary_structure
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import moroni as M
from reproduce import SEEDS,environment
from manuscript_tables import replace_table
RES=ROOT/'benchmarks/results'
TABLE_LABELS=dict(shape_original='tbl:shape',coverage='tbl:cov',reach='tbl:reach',
                  lut_original='tbl:lut_err',cpu_original='tbl:perf')

def table(name,columns,header,rows):
    text='\\begin{tabular}{'+columns+'}\\toprule\n'+header+' \\\\ \\midrule\n'
    text+='\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\\end{tabular}\n'
    replace_table(TABLE_LABELS[name],text)

def metrics(labels):
    cutoff=max(1,int(np.ceil(labels.size*.0001)));sol=[];comps=[]
    for c in np.unique(labels[labels>=0]):
        y,x=np.nonzero(labels==c)
        if len(x)<cutoff:continue
        corners=np.vstack([np.c_[x+dx,y+dy] for dx,dy in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5))])
        sol.append(len(x)/ConvexHull(corners).volume)
        cc,_=label(labels==c,generate_binary_structure(2,2))
        comps.append(int(np.sum(np.bincount(cc.ravel())[1:]>=cutoff)))
    return dict(solidity=float(np.median(sol)),low_solidity_pct=float(100*np.mean(np.array(sol)<.98)),
                mean_components=float(np.mean(comps)),disconnected_pct=float(100*np.mean(np.array(comps)>1)),
                valid_cells=len(sol),minimum_component_pixels=cutoff)

def run():
    control=json.loads((RES/'control.json').read_text());shape=[]
    for row in control['rows']:
        n=row['sides'];unclipped=[]
        for seed in SEEDS:
            f=M.make_cells(sides=n,radius=.62,jitter=.05,seed=seed);x,y=M.grid(320,200,f.bbox)
            raw=M.render_binned(f,x,y,analytic=True)
            unclipped.append(M.shape_fidelity(raw,f,x,y,min_pixels=16))
        means=[row['means'][0],row['means'][1],float(np.mean(unclipped)),row['means'][3]]
        shape.append(dict(sides=n,means=means,unclipped_per_seed=unclipped))
    table('shape_original','lrrrr',r'$n_c$ & Voronoi & Vor.+Clip & MORONI & MOR.+Clip',
          [[str(r['sides'])]+[f'{v:.3f}' for v in r['means']] for r in shape])
    margins=[-.10,-.05,0,.05,.12,.25];coverage=[]
    for seed in SEEDS:
        f=M.make_cells(seed=seed);x,y=M.grid(320,200,f.bbox);raw=M.render_binned(f,x,y,analytic=True)
        coverage.append([float((M.apply_margin(raw,f,x,y,m,True)>=0).mean()) for m in margins])
    means=np.mean(coverage,axis=0)
    table('coverage','lrrrrrr','Margin $m$ & '+' & '.join(f'${m:.2f}$' for m in margins),
          [[r'Coverage (\%)']+[f'{100*v:.1f}' for v in means]])
    f=M.make_cells();x,y=M.grid(320,200,f.bbox);reach=[]
    for kind in ('Euclidean','Power','Manhattan','Chebyshev','Apollonius'):
        best=np.full(x.shape,np.inf);ids=np.zeros(x.shape,np.int32)
        for c,ctr in enumerate(f.centers):
            dx=x-ctr[0];dy=y-ctr[1];d=np.hypot(dx,dy)
            e={'Euclidean':d,'Power':d*d-f.radii[c]**2,'Manhattan':abs(dx)+abs(dy),
               'Chebyshev':np.maximum(abs(dx),abs(dy)),'Apollonius':d-f.radii[c]}[kind]
            take=e<best;best[take]=e[take];ids[take]=c
        reach.append(dict(name=kind,**metrics(ids)))
    for g in (0,.05,.1,.2,.35):
        reach.append(dict(name=f'MORONI $\\gamma={g:.2f}$',**metrics(M.render_binned(f,x,y,gamma=g,analytic=True))))
    table('reach','lrrrr',r'Generator & Solidity & $S<.98$ (\%) & Mean comp. & Multiple comp. (\%)',
          [[r['name'],f"{r['solidity']:.3f}",f"{r['low_solidity_pct']:.1f}",f"{r['mean_components']:.2f}",f"{r['disconnected_pct']:.1f}"] for r in reach])
    lut=json.loads((RES/'lut.json').read_text())['rows']
    for r in lut:
        f=M.make_cells(lut_size=r['size']);rng=np.random.default_rng(42);sq=[]
        for c in range(f.n):
            b=f.lut_radius[c];dx=rng.uniform(-2*b,2*b,4000);dy=rng.uniform(-2*b,2*b,4000);p=f.polys[c]-f.centers[c]
            err=M._sample_lut(dx,dy,f.luts[c],b,p)-M._signed_distance(dx,dy,p);sq.append(err*err)
        r['rmse']=float(np.sqrt(np.concatenate(sq).mean()))
    table('lut_original','rrrrr',r'$L$ & MAE & RMSE & Max. error & Labels (\%)',
          [[str(r['size']),f"{r['mae']:.6f}",f"{r['rmse']:.6f}",f"{r['maximum']:.5f}",f"{r['mismatch_pct']:.4f}"] for r in lut])
    bench=json.loads((RES/'bench.json').read_text())['rows'];rows=[]
    for r in bench:
        t=r['median_ms'];rows.append([f"{r['width']}$\\times${r['height']}",f"{t['voronoi']:.1f}",f"{t['hybrid_binned']:.1f}",f"{t['hybrid_full']:.1f}",
            f"{t['hybrid_full']/t['hybrid_binned']:.1f}$\\times$ / {t['voronoi']/t['hybrid_binned']:.2f}$\\times$"])
    table('cpu_original','lrrrr',r'Pixels & Voronoi & MORONI & Full & Full/M. / Vor./M.',rows)
    result=dict(environment=environment(),seeds=SEEDS,resolution=[320,200],shape_columns=['Voronoi','Voronoi+clip','MORONI','MORONI+clip'],
        shape_field=dict(radius=.62,jitter=.05,gamma=.1,min_target_pixels=16),shape=shape,
        coverage=dict(margins=margins,per_seed=coverage,mean=means.tolist(),std=np.std(coverage,axis=0,ddof=1).tolist()),
        raster=dict(seed=22,clipping=False,connectivity=8,minimum_domain_area=.0001,weights='Power R_c^2, Apollonius R_c',rows=reach),lut=lut)
    (RES/'paper_tables.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print('Original table layout regenerated:',len(shape),'shape rows;',len(reach),'raster rows; coverage, LUT and CPU tables.')
    print('Coverage:',means.tolist());print('Shape:',[r['means'] for r in shape]);print('Phase .35:',reach[-1])

if __name__=='__main__':run()
