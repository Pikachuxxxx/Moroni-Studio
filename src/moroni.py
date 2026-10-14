"""Finite-field polygon-guided textures. Analytical distance defines the method;
hybrid LUT sampling is an approximation. Binning changes only candidate search.
Real-arithmetic bounds and measured floating-point agreement are distinguished.
"""
from dataclasses import dataclass
import numpy as np

__all__ = ['CellField','make_cells','grid','render_reference','render_pruned',
           'render_binned','BinnedRenderer','apply_margin','shape_fidelity',
           'voronoi_kdtree','colorize','boundary_density','geometry_metrics']
BACKGROUND = -1


def _prng(seed):
    """Mulberry32, shared with JavaScript, with explicit unsigned overflow."""
    s = int(seed) & 0xffffffff
    def draw():
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xffffffff
        t = ((s ^ (s >> 15)) * (1 | s)) & 0xffffffff
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xffffffff)) & 0xffffffff) ^ t
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296
    return draw


@dataclass
class CellField:
    centers: np.ndarray
    polys: list
    lattice: np.ndarray
    luts: np.ndarray
    lut_radius: np.ndarray
    radii: np.ndarray
    rows: int
    cols: int
    spacing: float

    @property
    def n(self): return len(self.centers)
    @property
    def bbox(self):
        s = self.spacing
        return (-s/2, (self.cols-.5)*s, -s/2, (self.rows-.5)*s)
    @property
    def lut_depth(self): return np.maximum(-self.luts.min(axis=(1, 2)), 0)
    @property
    def lut_error(self): return np.sqrt(2)*self.lut_radius/(self.luts.shape[-1]-1)


def _signed_distance(x, y, poly):
    x,y = np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
    distance = np.full(x.shape,np.inf)
    inside = np.zeros(x.shape,bool)
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        vx,vy = b-a
        t = np.clip(((x-a[0])*vx+(y-a[1])*vy)/max(vx*vx+vy*vy,1e-300),0,1)
        distance = np.minimum(distance,np.hypot(x-a[0]-t*vx,y-a[1]-t*vy))
        if vy != 0:
            inside ^= ((a[1]>y)!=(b[1]>y)) & (x<a[0]+(y-a[1])*vx/vy)
    return np.where(inside,-distance,distance)


def make_cells(rows=8,cols=12,spacing=.9,radius=.5,jitter=.35,
               seed=22,lut_size=72,sides=(5,9)):
    if (rows<1 or cols<1 or spacing<=0 or radius<=0 or
        not np.isfinite([spacing,radius,jitter]).all() or jitter<0 or lut_size<2):
        raise ValueError('Positive dimensions/radii and finite nonnegative jitter required')
    lo,hi = (sides,sides+1) if isinstance(sides,int) else sides
    if not 3<=lo<hi<=13: raise ValueError('Polygon sides must be in [3,12]')
    draw = _prng(seed)
    centers,polys,lattice,radii = [],[],[],[]
    for row in range(rows):
        for col in range(cols):
            ctr = np.array([(col+.26*(draw()-.5))*spacing,(row+.26*(draw()-.5))*spacing])
            n = lo+int(draw()*(hi-lo)); r = radius*(.9+.2*draw())
            # Circular vertex order and gaps < pi: simple convex polygons containing center.
            j = min(jitter,.45*np.pi/n)
            angles = np.array([2*np.pi*k/n+(2*draw()-1)*j for k in range(n)])
            polys.append(ctr+r*np.c_[np.cos(angles),np.sin(angles)])
            centers.append(ctr); lattice.append((row,col)); radii.append(r)
    luts=[]
    for c,p,r in zip(centers,polys,radii):
        ax=np.linspace(-1.5*r,1.5*r,lut_size); x,y=np.meshgrid(ax,ax)
        luts.append(_signed_distance(x,y,p-c))
    return CellField(np.array(centers),polys,np.array(lattice),np.array(luts),
                     1.5*np.array(radii),np.array(radii),rows,cols,spacing)


def grid(w,h,bbox):
    """Pixel centers, y increasing downwards, matching browser readback."""
    x0,x1,y0,y1=bbox
    return np.meshgrid(x0+(np.arange(w)+.5)*(x1-x0)/w,y0+(np.arange(h)+.5)*(y1-y0)/h)


def _sample_lut(dx,dy,lut,r,poly=None):
    x,y=np.broadcast_arrays(np.asarray(dx,float),np.asarray(dy,float)); n=lut.shape[-1]
    gx=np.clip((x+r)*(n-1)/(2*r),0,n-1); gy=np.clip((y+r)*(n-1)/(2*r),0,n-1)
    ix=np.minimum(gx.astype(int),n-2); iy=np.minimum(gy.astype(int),n-2)
    fx,fy=gx-ix,gy-iy
    value=(1-fx)*(1-fy)*lut[iy,ix]+fx*(1-fy)*lut[iy,ix+1]+(1-fx)*fy*lut[iy+1,ix]+fx*fy*lut[iy+1,ix+1]
    outside=(abs(x)>r)|(abs(y)>r)
    if outside.any():
        if poly is None: raise ValueError('Outside LUT: analytical polygon required')
        value=np.array(value,copy=True)
        value[outside]=_signed_distance(x[outside],y[outside],poly)
    return value


def _sdf(field,cid,x,y,analytic=False):
    dx,dy=x-field.centers[cid,0],y-field.centers[cid,1]
    poly=field.polys[cid]-field.centers[cid]
    return (_signed_distance(dx,dy,poly) if analytic else
            _sample_lut(dx,dy,field.luts[cid],field.lut_radius[cid],poly))


def _validate(alpha,beta,gamma,kappa,lam):
    if not np.isfinite([alpha,beta,gamma,kappa,lam]).all(): raise ValueError('Finite parameters required')
    if min(alpha,beta,gamma)<0 or int(kappa)!=kappa or not 1<=kappa<=6 or lam<0:
        raise ValueError('Nonnegative weights/frequency and integer harmonic 1..6 required')


def _phase_atan2(dx,dy,dc,kappa,lam):
    # Center convention, not a direction-independent continuous limit.
    return np.where(dc<1e-6,0.,1-np.cos(kappa*np.arctan2(dy,dx)+lam*dc))


def _energy(f,c,x,y,a,b,g,k,l,analytic):
    dx,dy=x-f.centers[c,0],y-f.centers[c,1]; d=np.hypot(dx,dy)
    return a*d+b*_sdf(f,c,x,y,analytic)+g*_phase_atan2(dx,dy,d,k,l)


def render_reference(field,xv,yv,alpha=.58,beta=.32,gamma=.1,kappa=2,lam=1.,analytic=False,return_energy=False):
    _validate(alpha,beta,gamma,kappa,lam)
    best=np.full(xv.shape,np.inf); labels=np.zeros(xv.shape,np.int32)
    for c in range(field.n):
        e=_energy(field,c,xv,yv,alpha,beta,gamma,kappa,lam,analytic)
        win=e<best; labels[win]=c; best[win]=e[win]
    return (labels,best) if return_energy else labels


class BinnedRenderer:
    """Finite ring search with a query-specific stopping certificate.

    No ring limit or cached query bins depend on constructor weights. Reuse
    with new weights, harmonics, domains, or coordinates is supported.
    """
    def __init__(self,field,alpha=.58,beta=.32,gamma=.1,kappa=2,lam=1.,analytic=False):
        self.f=field
        self.defaults=dict(alpha=alpha,beta=beta,gamma=gamma,kappa=kappa,lam=lam,analytic=analytic)

    def render(self,xv,yv,return_stats=False,**overrides):
        p=self.defaults|overrides
        unknown=p.keys()-{'alpha','beta','gamma','kappa','lam','analytic'}
        if unknown: raise TypeError(f'Unknown parameters: {unknown}')
        a,b,g,k,l=[p[n] for n in ('alpha','beta','gamma','kappa','lam')]
        _validate(a,b,g,k,l)
        f=self.f; s=f.spacing; x,y=xv.ravel(),yv.ravel()
        c0=np.clip(np.floor(x/s+.5).astype(int),0,f.cols-1)
        r0=np.clip(np.floor(y/s+.5).astype(int),0,f.rows-1)
        best=np.full(x.size,np.inf); labels=np.full(x.size,f.n,np.int32)
        active=np.ones(x.size,bool); stats=[]; evaluations=0
        eps=0 if p['analytic'] else float(f.lut_error.max())
        bound_radius=float(f.radii.max())+eps
        for ring in range(max(f.rows,f.cols)):
            idx=np.flatnonzero(active)
            if not len(idx): break
            for dr in range(-ring,ring+1):
                for dc in range(-ring,ring+1):
                    if max(abs(dr),abs(dc))!=ring: continue
                    rr,cc=r0[idx]+dr,c0[idx]+dc
                    valid=(rr>=0)&(rr<f.rows)&(cc>=0)&(cc<f.cols)
                    ii=idx[valid]; ids=rr[valid]*f.cols+cc[valid]
                    for cid in np.unique(ids):
                        q=ii[ids==cid]
                        d=np.hypot(x[q]-f.centers[cid,0],y[q]-f.centers[cid,1])
                        lower=(a+b)*d-b*(f.radii[cid]+eps)
                        guard=1e-12*(1+abs(lower)+np.where(np.isfinite(best[q]),abs(best[q]),0))
                        q=q[lower<=best[q]+guard]
                        if not len(q): continue
                        e=_energy(f,int(cid),x[q],y[q],a,b,g,k,l,p['analytic'])
                        win=(e<best[q])|((e==best[q])&(cid<labels[q]))
                        labels[q[win]]=cid; best[q[win]]=e[win]; evaluations+=len(q)
            # Remaining sites lie in four strips. Half-plane distances remain
            # conservative for queries outside the finite field's bounding box.
            distances=np.stack([
                np.where(c0[idx]-ring>0,np.maximum(x[idx]-(c0[idx]-ring-1+.13)*s,0),np.inf),
                np.where(c0[idx]+ring+1<f.cols,np.maximum((c0[idx]+ring+1-.13)*s-x[idx],0),np.inf),
                np.where(r0[idx]-ring>0,np.maximum(y[idx]-(r0[idx]-ring-1+.13)*s,0),np.inf),
                np.where(r0[idx]+ring+1<f.rows,np.maximum((r0[idx]+ring+1-.13)*s-y[idx],0),np.inf)])
            dmin=distances.min(0); remaining=np.isfinite(dmin)
            lower=np.full(len(idx),np.inf)
            lower[remaining]=(a+b)*dmin[remaining]-b*bound_radius
            guard=1e-12*(1+abs(best[idx])+np.where(remaining,abs(lower),0))
            active[idx]=remaining&(lower<=best[idx]+guard)
            stats.append((ring,len(idx),int(active.sum())))
        assert not active.any(), 'Finite search must exhaust or certify every pixel'
        result=labels.reshape(xv.shape)
        return (result,dict(pixels_per_ring=stats,energy_evaluations=evaluations,
                            mean_candidates=evaluations/x.size)) if return_stats else result


def render_binned(field,xv,yv,**kw): return BinnedRenderer(field).render(xv,yv,**kw)
def render_pruned(field,xv,yv,**kw): return render_binned(field,xv,yv,**kw)


def apply_margin(labels,field,xv,yv,margin,analytic=False):
    if margin is None: return labels.copy()
    if not np.isfinite(margin): raise ValueError('Finite margin or None required')
    out=labels.copy()
    for c in np.unique(labels[labels>=0]):
        mask=labels==c
        out[mask]=np.where(_sdf(field,int(c),xv[mask],yv[mask],analytic)>margin,-1,c)
    return out


def shape_fidelity(labels,field,xv,yv,min_pixels=50):
    """Target-based eligibility shared across methods; empty output scores zero."""
    values=[]
    for c in range(field.n):
        target=_sdf(field,c,xv,yv,True)<=0
        if target.sum()<min_pixels: continue
        region=labels==c; union=(region|target).sum()
        values.append((region&target).sum()/union if union else 0)
    return float(np.mean(values)) if values else 0.


def voronoi_kdtree(field,xv,yv):
    from scipy.spatial import cKDTree
    return cKDTree(field.centers).query(np.c_[xv.ravel(),yv.ravel()],workers=1)[1].reshape(xv.shape).astype(np.int32)


def geometry_metrics(labels,min_component_fraction=.0001,connectivity=8):
    """Descriptive raster metrics, not continuous topology certificates."""
    from scipy.ndimage import label,generate_binary_structure
    from scipy.spatial import ConvexHull
    solidities=[]; components=[]
    threshold=max(1,int(np.ceil(labels.size*min_component_fraction)))
    for c in np.unique(labels[labels>=0]):
        ys,xs=np.nonzero(labels==c)
        if len(xs)<threshold: continue
        corners=np.vstack([np.c_[xs+dx,ys+dy] for dx,dy in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5))])
        solidities.append(len(xs)/ConvexHull(corners).volume)
        lab,_=label(labels==c,generate_binary_structure(2,2 if connectivity==8 else 1))
        components.append(int(np.sum(np.bincount(lab.ravel())[1:]>=threshold)))
    return dict(median_solidity=float(np.median(solidities)) if solidities else None,
                disconnected_pct=100*float(np.mean(np.array(components)>1)) if components else 0,
                valid_cells=len(components),threshold_pixels=threshold,connectivity=connectivity)


def cell_colors(n,seed=33):
    from matplotlib.colors import hsv_to_rgb
    draw=_prng(seed)
    hues=np.array([(i/n+.04*(draw()-.5))%1 for i in range(n)])
    saturation=np.array([.46+.24*draw() for _ in range(n)])
    value=np.array([.68+.22*draw() for _ in range(n)])
    return hsv_to_rgb(np.c_[hues,saturation,value])


def colorize(labels,seed=33,edge_dark=.27,n_cells=None):
    n=n_cells or max(1,int(labels.max())+1)
    colors=cell_colors(n,seed)
    image=colors[np.maximum(labels,0)]; image[labels<0]=.07
    return image


def boundary_density(labels):
    return float(((labels[1:,:]!=labels[:-1,:]).sum()+(labels[:,1:]!=labels[:,:-1]).sum())/labels.size)
