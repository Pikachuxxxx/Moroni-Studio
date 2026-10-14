"""Authoritative experiment pipeline; JSON and LaTeX tables come from one run.
Run from repo root: python benchmarks/reproduce.py [all|verify|control|bench|figures]
"""
import json,sys,time,platform,hashlib,argparse
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import moroni as M
RES=ROOT/'benchmarks'/'results';FIG=ROOT/'paper'/'figures'
for p in (RES,FIG):p.mkdir(exist_ok=True)
SEEDS=[7,22,42,100,101,102,103,104]


def write(name,obj):
    (RES/(name+'.json')).write_text(json.dumps(obj,indent=2),newline='\n')


def environment():
    return dict(platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),
                numpy=np.__version__,scipy=scipy.__version__,source_sha256=hashlib.sha256((ROOT/'src/moroni.py').read_bytes()).hexdigest())


def fixture():
    params=dict(rows=8,cols=12,spacing=.9,radius=.5,jitter=.35,sides=0,seed=22)
    f=M.make_cells();x,y=M.grid(96,64,f.bbox)
    labels=M.render_reference(f,x,y,analytic=True)
    rec=dict(params=params,centers=f.centers.tolist(),polys=[p.tolist() for p in f.polys],
             colors=M.cell_colors(f.n).tolist(),width=96,height=64,labels=labels.ravel().tolist())
    (ROOT/'tests'/'field_fixture.json').write_text(json.dumps(rec),newline='\n')


def verify():
    rng=np.random.default_rng(892);cases=[]
    for i in range(60):
        n=int(rng.integers(3,9)); seed=int(rng.integers(0,10000))
        params=dict(alpha=float(rng.uniform(.1,1.2)),beta=float(rng.uniform(0,1.5)),
                    gamma=float(rng.uniform(0,.45)),kappa=int(rng.integers(1,7)),lam=float(rng.uniform(.2,4)))
        field_params=dict(rows=4,cols=6,sides=n,radius=float(rng.uniform(.3,.85)),jitter=float(rng.uniform(0,.6)),seed=seed)
        f=M.make_cells(**field_params)
        x,y=M.grid(96,64,f.bbox)
        if i%10==0:x=x-2;y=y+2
        renderer=M.BinnedRenderer(f)
        for analytic in (False,True):
            ref=M.render_reference(f,x,y,analytic=analytic,**params)
            got,stats=renderer.render(x,y,return_stats=True,analytic=analytic,**params)
            mismatch=int((ref!=got).sum())
            cases.append(dict(case=i,seed=seed,sides=n,field_params=field_params,params=params,
                              query_offset=[-2,2] if i%10==0 else [0,0],analytic=analytic,mismatch=mismatch,
                              mean_candidates=stats['mean_candidates']))
            assert mismatch==0,(i,params,mismatch)
    out=dict(environment=environment(),cases=cases,total_pixels=len(cases)*96*64,total_mismatches=sum(c['mismatch'] for c in cases))
    write('correctness',out);fixture()
    print('Correctness:',len(cases),'cases,',out['total_pixels'],'pixels, 0 mismatches',flush=True)


def controls():
    rows=[]
    for n in (3,4,5,6,8):
        measurements=[]
        for seed in SEEDS:
            f=M.make_cells(sides=n,radius=.62,jitter=.05,seed=seed);x,y=M.grid(320,200,f.bbox)
            vor=M.voronoi_kdtree(f,x,y)
            polygon=M.render_binned(f,x,y,gamma=0,analytic=True)
            full=M.render_binned(f,x,y,gamma=.1,analytic=True)
            warp=.12*f.spacing
            warped=M.voronoi_kdtree(f,x+warp*np.sin(3*y/f.spacing),y+warp*np.sin(3*x/f.spacing))
            masks=[vor,M.apply_margin(vor,f,x,y,0,True),M.apply_margin(polygon,f,x,y,0,True),
                   M.apply_margin(full,f,x,y,0,True),M.apply_margin(warped,f,x,y,0,True)]
            measurements.append(dict(seed=seed,iou=[M.shape_fidelity(m,f,x,y,min_pixels=16) for m in masks],
                                     coverage=[float((m>=0).mean()) for m in masks]))
        data=np.array([m['iou'] for m in measurements]);delta=data[:,2]-data[:,1]
        rows.append(dict(sides=n,measurements=measurements,means=data.mean(0).tolist(),
                         std=data.std(0,ddof=1).tolist(),paired_delta=float(delta.mean()),delta_std=float(delta.std(ddof=1))))
        print('Control',n,rows[-1]['means'],flush=True)
    out=dict(environment=environment(),seeds=SEEDS,resolution=[320,200],min_target_pixels=16,
             columns=['Voronoi','Voronoi+clip','MORONI gamma=0 +clip','MORONI gamma=.1 +clip','warped Voronoi+clip'],rows=rows)
    write('control',out)
    # Distance error on actual query domain, including outside-LUT queries.
    error=[]
    for size in (36,72,144):
        f=M.make_cells(lut_size=size);x,y=M.grid(320,200,f.bbox)
        exact=M.render_reference(f,x,y,analytic=True);approx=M.render_reference(f,x,y)
        abs_errors=[];max_err=0;max_bound=0
        rng=np.random.default_rng(42)
        for c in range(f.n):
            r=f.lut_radius[c];dx=rng.uniform(-2*r,2*r,4000);dy=rng.uniform(-2*r,2*r,4000)
            p=f.polys[c]-f.centers[c];a=M._signed_distance(dx,dy,p);b=M._sample_lut(dx,dy,f.luts[c],r,p)
            diff=abs(a-b);abs_errors.append(float(diff.mean()));max_err=max(max_err,float(diff.max()));max_bound=max(max_bound,float(f.lut_error[c]))
            assert diff.max()<=f.lut_error[c]+1e-12
        error.append(dict(size=size,mae=float(np.mean(abs_errors)),maximum=max_err,bound=max_bound,
                          mismatch_pixels=int((exact!=approx).sum()),mismatch_pct=float((exact!=approx).mean()*100)))
    write('lut',dict(query_seed=42,queries_per_polygon=4000,polygons=96,field_resolution=[320,200],rows=error))
    # Raster topology stability: fixed domain-area threshold and explicit clipping.
    topo=[];f=M.make_cells()
    for w,h in ((320,200),(640,400),(960,600)):
        x,y=M.grid(w,h,f.bbox);raw=M.render_binned(f,x,y,gamma=.35,analytic=True)
        for margin in (None,0):
            labels=M.apply_margin(raw,f,x,y,margin,True)
            for conn in (4,8):
                topo.append(dict(width=w,height=h,margin=margin,**M.geometry_metrics(labels,connectivity=conn)))
    write('raster',dict(seed=22,gamma=.35,minimum_component_domain_fraction=.0001,rows=topo))


def bench():
    f=M.make_cells();rows=[]
    for w,h in ((320,200),(640,400),(960,600)):
        x,y=M.grid(w,h,f.bbox);r=M.BinnedRenderer(f)
        methods={'voronoi':lambda:M.voronoi_kdtree(f,x,y),'hybrid_full':lambda:M.render_reference(f,x,y),
                 'hybrid_binned':lambda:r.render(x,y),'analytic_binned':lambda:r.render(x,y,analytic=True)}
        for fn in methods.values():fn()
        samples={name:[] for name in methods}
        for repeat in range(5):
            for name,fn in methods.items():
                start=time.perf_counter();fn();samples[name].append((time.perf_counter()-start)*1000)
        means={name:float(np.median(v)) for name,v in samples.items()}
        rows.append(dict(width=w,height=h,median_ms=means,samples_ms=samples));print('Timing',w,means,flush=True)
    write('bench',dict(environment=environment(),warmup=1,rounds=5,seed=22,threads='KD-tree workers=1; NumPy vectorized CPU',rows=rows))


def figures():
    from paper_tables import run
    from paper_figures import figures as refresh_figures
    run()
    refresh_figures()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['all','verify','control','bench','figures'],default='all',nargs='?');a=p.parse_args()
    tasks={'verify':verify,'control':controls,'bench':bench,'figures':figures}
    for name,fn in tasks.items():
        if a.command in ('all',name):fn()
