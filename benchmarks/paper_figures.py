"""Refresh the original paper's figures in their existing panel layouts."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import moroni as M
FIG=ROOT/'paper/figures';RES=ROOT/'benchmarks/results'
plt.rcParams.update({'font.size':22,'axes.titlesize':23,'axes.labelsize':21,'legend.fontsize':17})

def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/name,dpi=150,bbox_inches='tight');plt.close(fig)
def show(ax,labels,f,title):
    ax.imshow(M.colorize(labels,n_cells=f.n));ax.set_title(title);ax.axis('off')
def render(f,margin=None,**kw):
    x,y=M.grid(620,420,f.bbox);raw=M.render_binned(f,x,y,analytic=True,**kw)
    return M.apply_margin(raw,f,x,y,margin,True)

def figures():
    f=M.make_cells(sides=3,radius=.62,jitter=.05,seed=7,rows=5,cols=7);x,y=M.grid(1000,720,f.bbox)
    raw=M.render_binned(f,x,y,gamma=0,analytic=True)
    fig,ax=plt.subplots(1,3,figsize=(20,5))
    for a,l,t in zip(ax,[M.voronoi_kdtree(f,x,y),raw,M.apply_margin(raw,f,x,y,0,True)],
        ['Voronoi','Polygon energy, full partition','Polygon energy + clipping']):show(a,l,f,t)
    save(fig,'shape_control.png')
    f=M.make_cells();x,y=M.grid(700,450,f.bbox);raw=M.render_binned(f,x,y,analytic=True)
    dc=np.zeros(x.shape);sc=dc.copy();phase=dc.copy()
    for c in range(f.n):
        take=raw==c;dx=x[take]-f.centers[c,0];dy=y[take]-f.centers[c,1];d=np.hypot(dx,dy)
        dc[take]=d;sc[take]=M._signed_distance(dx,dy,f.polys[c]-f.centers[c]);phase[take]=M._phase_atan2(dx,dy,d,2,1)
    fig,ax=plt.subplots(1,4,figsize=(22,5))
    for a,data,title,cm in zip(ax[:3],[dc,sc,phase],['Center distance $d_c$','Signed polygon distance $s_c$','Phase $1-\\cos(2\\theta+d)$'],['magma','coolwarm','magma']):
        a.imshow(data,cmap=cm);a.set_title(title);a.axis('off')
    show(ax[3],raw,f,'Assignment argmin $E_c$');save(fig,'energy_decomposition.png')
    for filename,title,key,values,margin in [
        ('control_shape.png','Vertex count: target shape','sides',[3,4,5,6,8],0),
        ('control_size.png','Radius: target size','radius',[.30,.42,.54,.66,.78],.02),
        ('control_irregularity.png','Angular jitter (subject to validity cap)','jitter',[0,.15,.30,.45,.60],.02)]:
        fig,ax=plt.subplots(1,5,figsize=(23,4.4))
        for a,v in zip(ax,values):
            kw=dict(radius=.6,jitter=.15,seed=7);kw[key]=v;f=M.make_cells(**kw)
            show(a,render(f,margin,gamma=0),f,f'{v:g}')
        fig.suptitle(title,y=1.02);save(fig,filename)
    for filename,title,key,values,defaults in [
        ('control_mortar.png','Coverage margin: retained area','margin',[-.08,-.03,.02,.08,.18],{}),
        ('control_fracture.png','Phase weight: boundary variation','gamma',[0,.08,.16,.26,.35],dict(margin=.05)),
        ('control_fidelity.png','Polygon weight: assignment influence','beta',[0,.2,.5,.9,1.4],dict(margin=.02,gamma=0))]:
        f=M.make_cells(radius=.58,jitter=.18,seed=7);fig,ax=plt.subplots(1,5,figsize=(23,4.4))
        for a,v in zip(ax,values):
            p=dict(defaults);p[key]=v;show(a,render(f,**p),f,f'{v:.2f}')
        fig.suptitle(title,y=1.02);save(fig,filename)
    presets=[('Cracked mud',7,.60,.40,.32,.30,.05,2,1),('Stone wall',5,.62,.12,.90,.02,.03,2,1),
             ('Ceramic tile',4,.60,.02,1.20,0,.01,2,1),('Shattered glass',4,.70,.50,.32,.35,.08,3,1.8)]
    fig,ax=plt.subplots(1,4,figsize=(20.8,4.8))
    for a,(name,n,r,j,b,g,m,k,lam) in zip(ax,presets):
        f=M.make_cells(sides=n,radius=r,jitter=j,seed=22);show(a,render(f,m,beta=b,gamma=g,kappa=k,lam=lam),f,name)
    save(fig,'presets.png')
    f=M.make_cells();x,y=M.grid(700,450,f.bbox);fig,ax=plt.subplots(1,2,figsize=(18,5.8))
    show(ax[0],M.voronoi_kdtree(f,x,y),f,'Voronoi');show(ax[1],M.apply_margin(M.render_binned(f,x,y,analytic=True),f,x,y,.05,True),f,'MORONI + coverage margin');save(fig,'voronoi_vs_moroni.png')
    data=json.loads((RES/'paper_tables.json').read_text());rows=data['shape'];ns=[r['sides'] for r in rows]
    fig,ax=plt.subplots(1,3,figsize=(19,5))
    for idx,title,style in [(0,'Voronoi','^:'),(1,'Voronoi + clipping','d--'),(2,'MORONI, full partition','s--'),(3,'MORONI + clipping','o-')]:
        ax[0].plot(ns,[r['means'][idx] for r in rows],style,label=title,lw=2)
    ax[0].set(xlabel='Vertex count',ylabel='Mean target IoU',title='Matched shape-fidelity comparison',ylim=(0,1.02));ax[0].legend()
    cov=data['coverage'];ax[1].errorbar(cov['margins'],100*np.array(cov['mean']),yerr=100*np.array(cov['std']),fmt='o-',capsize=3)
    ax[1].set(xlabel='Coverage margin (world distance)',ylabel='Retained pixels (%)',title='Coverage, mean and seed SD')
    rr=data['raster']['rows'][-5:];g=[0,.05,.1,.2,.35]
    ax[2].plot(g,[r['solidity'] for r in rr],'o-',color='#8e44ad',label='MORONI solidity')
    ax[2].axhline(data['raster']['rows'][0]['solidity'],ls='--',color='black',label='Sampled Voronoi solidity')
    twins=ax[2].twinx();twins.plot(g,[r['disconnected_pct'] for r in rr],'s-',color='#c0392b');twins.set_ylabel('Cells with multiple components (%)',color='#c0392b')
    ax[2].set(xlabel='Phase weight',ylabel='Median solidity',title='Descriptive raster geometry');ax[2].legend(loc='lower left',fontsize=16)
    for a in ax:a.grid(alpha=.25)
    save(fig,'control_quantitative.png')
    rows=json.loads((RES/'bench.json').read_text())['rows'];idx=np.arange(len(rows));labels=[f"{r['width']}x{r['height']}" for r in rows]
    plt.rcParams.update({'font.size':26,'axes.titlesize':28,'axes.labelsize':27,'legend.fontsize':23})
    fig,ax=plt.subplots(1,2,figsize=(16,7.2))
    for key,title,style in [('voronoi','KD-tree Voronoi','o-'),('hybrid_binned','MORONI binned','^-'),('hybrid_full','Matching full','s-')]:
        ax[0].plot(idx,[r['median_ms'][key] for r in rows],style,label=title,lw=2)
    ax[0].set(yscale='log',ylabel='Median time (ms)',title='CPU runtime');ax[0].legend()
    ax[1].bar(idx-.2,[r['median_ms']['hybrid_full']/r['median_ms']['hybrid_binned'] for r in rows],.4,label='Full / binned')
    ax[1].bar(idx+.2,[r['median_ms']['voronoi']/r['median_ms']['hybrid_binned'] for r in rows],.4,label='Voronoi / binned')
    ax[1].axhline(1,color='black',ls='--');ax[1].set(ylabel='Ratio',title='Ratios from matching medians');ax[1].legend()
    for a in ax:a.set_xticks(idx,labels);a.grid(alpha=.25)
    save(fig,'performance.png');print('Refreshed existing figure filenames and panel layouts from the corrected code/results.')

if __name__=='__main__':figures()
